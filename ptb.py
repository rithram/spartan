#!/usr/bin/env python

import argparse
import os
import time
import sys
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import PennTreebank
from torchtext.data.functional import (
  generate_sp_model, load_sp_model, sentencepiece_numericalizer
)

print(f"Cuda available: {torch.cuda.is_available()}")
GPUE = torch.cuda.is_available()

from tf_model import TFClassifier

def batch_yield(dataset, sp_iterator, context_size, batch_size):
  X, y = [], []
  nline = 1
  for line in sp_iterator(dataset):
    nline += 1
    start = 0
    while start + context_size < len(line):
      X += [line[start: start + context_size]]
      y += [line[start + context_size]]
      start += 1
      if len(X) >= batch_size:
        yield (
          torch.tensor(X, dtype=torch.long).to('cuda'),
          torch.tensor(y, dtype=torch.long).to('cuda')
        )
        X, y = [], []
  yield (
    torch.tensor(X, dtype=torch.long).to('cuda'),
    torch.tensor(y, dtype=torch.long).to('cuda')
  )

def train_epoch(
    model: nn.Module,
    train_data,
    spiece_iter,
    nclass: int,
    context_size: int,
    crit,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    bsz: int,
    tbatches: int,
    epoch: int,
    log_interval: int = 100,
) -> None:
  model.train()  # turn on train mode
  total_loss = 0.
  start_time = time.time()

  for batch, (data, targets) in enumerate(
      batch_yield(train_data, spiece_iter, context_size, bsz)
  ):
    output = model(data)
    output_flat = output.view(-1, nclass)
    loss = crit(output_flat, targets)

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()

    total_loss += loss.item()
    if np.isnan(total_loss):
      print(f"Reached loss={total_loss} as batch {batch} of epoch {epoch} ... QUIT")
      sys.exit(0)
    if batch % log_interval == 0 and batch > 0:
      lr = sched.get_last_lr()[0]
      ms_per_batch = (time.time() - start_time) * 1000 / log_interval
      cur_loss = total_loss / log_interval
      print(
        f'| epoch {epoch:3d} | {batch:5d}/{tbatches:5d} batches | '
        f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
        f'loss {cur_loss:5.2f}'
      )
      total_loss = 0
      start_time = time.time()

def evaluate(
    model: nn.Module,
    train_data,
    spiece_iter,
    nclass: int,
    context_size: int,
    crit,
    bsz: int,
) -> (float, float):
  model.eval()  # turn on evaluation mode
  total_loss = 0.
  total_correct = 0
  total_examples = 0
  with torch.no_grad():
    for batch, (data, targets) in enumerate(
        batch_yield(train_data, spiece_iter, context_size, bsz)
    ):
      output = model(data)
      output_flat = output.view(-1, nclass)
      total_loss += (data.size(0) * crit(output_flat, targets).item())
      _, preds = torch.max(output, 1)
      total_correct += (targets == preds).sum().item()
      total_examples += bsz
  return (
    total_loss / total_examples,
    total_correct * 100.0 / total_examples
  )

# size lambdas
set_length = lambda iterator: len([item for item in iterator])
avg_length = lambda iterator: np.mean([len(item) for item in iterator]).astype(int)
qtl_length = lambda iterator, percentile: np.percentile(
  [len(item) for item in iterator], q=percentile
).astype(int)
max_length = lambda iterator: qtl_length(iterator, 100)
min_length = lambda iterator: qtl_length(iterator, 0)


def print_stats(diterator, tokenizer=None):
  it = lambda iterator: iterator if tokenizer is None else tokenizer(iterator)
  print(
    f"Total number of sentences: {set_length(it(diterator))} "
    f"[min: {min_length(it(diterator))}, "
    f"mean: {avg_length(it(diterator))}, "
    f"median: {qtl_length(it(diterator), 50)}, "
    f"max: {max_length(it(diterator))}]"
  )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-e', '--demb', help="Embedding dim", type=int, default=32
  )
  parser.add_argument(
    '-m', '--dmlp', help="MLP hidden dim", type=int, default=32
  )
  parser.add_argument(
    '-B', '--nblocks', help="# transformer blocks", type=int, default=4,
  )
  parser.add_argument(
    '-H', '--nheads', help="# attention heads per block", type=int, default=1,
  )
  mask_choices=['none', 'windowed', 'blklocal', 'topk', 'botk', 'rnd']
  NM, WM, BM, TM, MM, RM = 0, 1, 2, 3, 4, 5
  parser.add_argument(
    '-M', '--mask', help='Type of attention masks', choices=mask_choices,
    default=mask_choices[NM],
  )
  parser.add_argument(
    '-k', '--mask_size', help='NNZ per row in attention mask',
    type=int, default=-1,
  )
  parser.add_argument(
    '-d', '--dropout', help="Dropout", type=float, default=0.01,
  )
  parser.add_argument(
    '-l', '--init_lr', help="Initial learning rate", type=float, default=1.0,
  )
  parser.add_argument(
    '-D', '--lr_decay_rate', help="LR decay rate", type=float, default=0.9995,
  )
  parser.add_argument(
    '-b', '--bsz', help="Batch size", type=int, default=16,
  )
  parser.add_argument(
    '-E', '--nepochs', help="# epochs", type=int, default=10,
  )
  parser.add_argument(
    '-O', '--output_dir', help="Output directory", type=str, default="",
  )
  parser.add_argument(
    '-C', '--chkpt_dir', help="Checkpoint directory", type=str, default="",
  )
  parser.add_argument(
    '--seed', help="Experiment seed", type=int, default=1111,
  )
  mlp_activations = ['relu', 'gelu', 'mish']
  parser.add_argument(
    '--mlpa', help="Activation in MLP", choices=mlp_activations,
    default=mlp_activations[0]
  )
  parser.add_argument(
    '--sp_data', help="Sentence piece data for model training",
    type=str, default=""
  )
  parser.add_argument(
    '--sp_model_dir', help="Sentence piece model directory",
    type=str, required=True,
  )
  parser.add_argument(
    '--sp_vocab_size', help="Vocab size for sentence piece",
    type=int, required=True,
  )
  spchoices = ["unigram", "bpe", "char", "word"]
  parser.add_argument(
    '--sp_model_type', help="Model type for sentence piece",
    choices=spchoices, default=spchoices[0],
  )

  args = parser.parse_args()
  assert args.demb > 1
  assert args.dmlp >= args.demb
  assert args.nblocks >= 1
  assert args.nheads >= 1
  assert 0 < args.dropout < 1.
  assert args.init_lr > 0.
  assert 0 < args.lr_decay_rate < 1.
  assert args.bsz >= 1
  assert args.nepochs > 1
  assert args.output_dir == "" or os.path.exists(args.output_dir)
  assert args.chkpt_dir == "" or os.path.exists(args.chkpt_dir)

  SEED = args.seed

  ofile = 'ptb'
  for k, v in args.__dict__.items():
    if k in [
        "train", "val", "test",
        "output_dir", "chkpt_dir",
        "sp_data", "sp_model_dir",
    ]:
      continue
    ofile += f"__{k}:{v}"
  print(f"Param config:\n{ofile}")
  cfile_last = ofile + '_last.pt'
  cfile_best = ofile + '_best.pt'
  ofile += '.csv'
  if args.output_dir != "":
    ofile = os.path.join(args.output_dir, ofile)
    if os.path.exists(ofile):
      print("ERROR: Experiment already done -- EXITING")
      sys.exit(0)
    print(f"Will be saving results in \n{ofile}")
  if args.chkpt_dir != "":
    cfile_last = os.path.join(args.chkpt_dir, cfile_last)
    cfile_best = os.path.join(args.chkpt_dir, cfile_best)
    print(
      "Will be saving checkpoints in the following:\n"
      f"- Best epoch checkpoint: {cfile_best}\n"
      f"- Last epoch checkpoint: {cfile_last}\n"
    )
    assert not os.path.exists(cfile_last)
    assert not os.path.exists(cfile_best)

  # Set seeds for experiment
  RNG = np.random.RandomState(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  # Load the dataset
  train_data = PennTreebank(split='train')
  valid_data = PennTreebank(split='valid')

  print("Original data statistic")
  print("-"*40)
  print_stats(train_data)
  print_stats(valid_data)
  print("-"*40)

  # generate sentence piece model and tokenize
  PATH = args.sp_data
  VSIZE = args.sp_vocab_size
  MTYPE = args.sp_model_type
  assert os.path.exists(args.sp_model_dir)
  SPPREFIX = f"{args.sp_model_dir}/ptb_sp_{MTYPE}_{VSIZE}"
  SPMODEL = f"{SPPREFIX}.model"
  SPVOCAB = f"{SPPREFIX}.vocab"
  if not os.path.exists(SPMODEL):
    print(f"generating sentence piece model")
    assert os.path.exists(PATH)
    generate_sp_model(PATH, vocab_size=VSIZE, model_type=MTYPE, model_prefix=SPPREFIX)
  else:
    print(f"using existing sentence-piece model at {SPMODEL}")
  assert os.path.exists(SPMODEL)
  assert os.path.exists(SPVOCAB)

  sp_model = load_sp_model(open(SPMODEL, 'rb'))
  sp_id_gen = sentencepiece_numericalizer(sp_model)

  print("Sentence-piece tokenized data statistic")
  print("-"*40)
  print_stats(train_data, sp_id_gen)
  print_stats(valid_data, sp_id_gen)
  print("-"*40)

  CSIZE = avg_length(sp_id_gen(train_data))
  BSZ = args.bsz

  nbatches_per_epoch = 0
  for bidx, (X, y) in enumerate(batch_yield(train_data, sp_id_gen, CSIZE, BSZ)):
    assert 0 < X.shape[0] <= BSZ
    assert X.shape[1] == CSIZE
    assert y.shape[0] == X.shape[0]
    nbatches_per_epoch += 1

  print(
    f"Total number of batches (size: {BSZ}) in an epoch: {nbatches_per_epoch}"
    f" each with context size: {CSIZE}"
  )

  DEMB = args.demb
  DQKV = args.demb
  DMLP = args.dmlp
  NBLOCKS = args.nblocks
  NHEADS = args.nheads
  DROPOUT = args.dropout
  MASK = args.mask
  MASKSIZE = args.mask_size
  MLPA = args.mlpa
  INITLR = args.init_lr
  LRDRATE = args.lr_decay_rate
  NEPOCHS = args.nepochs

  chkpt_dir = args.chkpt_dir

  model = TFClassifier(
    ntoken=VSIZE,
    nclass=VSIZE,
    maxlen=CSIZE,
    d_emb=DEMB,
    d_qkv=DQKV,
    d_mlp=DMLP,
    nlayers=NBLOCKS,
    nheads=NHEADS,
    dropout=DROPOUT,
    mask_type=MASK,
    mask_size=MASKSIZE,
    cls_token=False,
    inv_temp=1.,
    ngtk=0,
    mlpa=MLPA,
  ).to('cuda')

  criterion = nn.CrossEntropyLoss()
  lr = INITLR  # learning rate
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=LRDRATE)

  train_stats = []
  train_cols = ["epoch", "tr-ce", "tr-ac", "va-ce", "va-ac"]
  current_best_val_acc = 0.0
  old_nepochs = 0
  new_nepochs = NEPOCHS

  print(
    f"Initializing epochs {old_nepochs}->{new_nepochs} with"
    f" current best validation accuracy {current_best_val_acc:.2f}..."
  )
  for epoch in range(old_nepochs + 1, new_nepochs + 1):
    epoch_start_time = time.time()
    epoch_stats = [epoch]
    train_epoch(
      model,
      train_data,
      sp_id_gen,
      VSIZE,
      CSIZE,
      criterion,
      optimizer,
      scheduler,
      BSZ,
      nbatches_per_epoch,
      epoch,
    )
    elapsed = time.time() - epoch_start_time
    teval = time.time()
    train_loss, train_acc = evaluate(
      model,
      train_data,
      sp_id_gen,
      VSIZE,
      CSIZE,
      criterion,
      BSZ * 16,
    )
    teval -= time.time()
    teval *= -1
    epoch_stats += [train_loss, train_acc]
    veval = time.time()
    val_loss, val_acc = evaluate(
      model,
      valid_data,
      sp_id_gen,
      VSIZE,
      CSIZE,
      criterion,
      BSZ * 16,
    )
    veval -= time.time()
    veval *= -1
    epoch_stats += [val_loss, val_acc]
    print('-' * 89)
    if val_acc > current_best_val_acc:
      current_best_val_acc = val_acc
      # Saving current best model
      if chkpt_dir != "":
        torch.save(model.state_dict(), cfile_best)
    print(
      f'| epoch {epoch:3d} |'
      f' {elapsed:5.0f}s/{teval:5.0f}s/{veval:5.0f}s |'
      f' t-loss: {train_loss:.2f} | v-loss: {val_loss:.2f} |'
      f' t-acc: {train_acc:.2f} | v-acc: {val_acc:.2f} |'
    )
    scheduler.step()
    train_stats += [epoch_stats]
    print('-' * 89)
    if args.output_dir != "":
      print(f"Saving training stats in {ofile} ...")
      cur = pd.DataFrame(
        train_stats, columns=train_cols
      )
      cur.to_csv(ofile, header=True, index=False)
  # Saving last epoch model weights + relevant expt info
  if args.chkpt_dir != "":
    print(f"Saving checkpoint of final state in\n{cfile_last}\n")
    torch.save({
      'cli_args_dict': args.__dict__,
      'train_stats': "" if args.output_dir == "" else ofile,
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'best_val_loss': current_best_val_acc,
      'chkpt_last': cfile_last,
      'chkpt_best': cfile_best,
    }, cfile_last)
  res_df = pd.DataFrame(train_stats, columns=train_cols)
  print(res_df.tail(10))
  print(f"Completed with best validation accuracy: {current_best_val_acc:.2f}")
