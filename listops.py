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
from torchtext.vocab import build_vocab_from_iterator

print(f"Cuda available: {torch.cuda.is_available()}")
GPUE = torch.cuda.is_available()

from tf_model import TFClassifier

# Function to tokenize input string
def ctok(text):
  return [
    c for c in text.replace('(', '').replace(')','').split(' ')
    if c != ''
  ]

# Token iterator for vocab generator
CLS = "[CLS]"
PAD = "[PAD]"
GTK = "GTK"
def yield_tokens(D, max_len=None, cls_token=True, ngtk=0):
  gtks = [] if ngtk == 0 else [f"[{GTK}{i+1}]" for i in range(ngtk)]
  for idx, row in D.iterrows():
    src = ctok(row['Source'])
    pad_len = (
      max(0, max_len - len(src) - int(cls_token))
      if max_len is not None else 0
    )
    if cls_token:
      src = [CLS] + src
    if max_len is None or pad_len == 0:
      yield src + gtks
    else:
      yield src + [PAD]*pad_len + gtks


def batch_yielder(N, B, RNG):
  idxs = np.arange(N)
  RNG.shuffle(idxs)
  num_batches = N // B
  for i in range(0, N-1, B):
    yield idxs[i: min(i+B, N-1)]

def train_epoch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    nclass: int,
    crit,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    bsz: int,
    RNG,
    epoch: int,
    log_interval: int = 25,
) -> None:
  model.train()  # turn on train mode
  total_loss = 0.
  start_time = time.time()

  num_batches = X.size(0) // bsz
  for batch, idx in enumerate(batch_yielder(X.size(0), bsz, RNG)):
    data = X[idx].to('cuda') if GPUE else X[idx]
    targets = y[idx].to('cuda') if GPUE else y[idx]
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
        f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
        f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
        f'loss {cur_loss:5.2f}'
      )
      total_loss = 0
      start_time = time.time()


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    nclass: int,
    crit,
    bsz: int,
) -> (float, float):
  model.eval()  # turn on evaluation mode
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for i in range(0, X.size(0) - 1, bsz):
      data = X[i:min(i + bsz, X.size(0) - 1)]
      # print(data.size(0))
      targets = y[i:min(i + bsz, y.size(0) - 1)]
      if GPUE:
        data = data.to('cuda')
        targets = targets.to('cuda')
      output = model(data)
      output_flat = output.view(-1, nclass)
      total_loss += (data.size(0) * crit(output_flat, targets).item())
      _, preds = torch.max(output, 1)
      total_correct += (targets == preds).sum().item()
      # break
  return (
    total_loss / X.size(0),
    total_correct * 100.0 / X.size(0)
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-T', '--train', help='Training data', type=str, required=True,
  )
  parser.add_argument(
    '-V', '--val', help='Validation data', type=str, required=True,
  )
  parser.add_argument(
    '-t', '--test', help='Test data', type=str, default=None,
  )
  parser.add_argument(
    '-e', '--demb', help="Embedding dim", type=int, default=128
  )
  parser.add_argument(
    '-m', '--dmlp', help="MLP hidden dim", type=int, default=128
  )
  parser.add_argument(
    '-B', '--nblocks', help="# transformer blocks", type=int, default=-1,
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
    "--nocls", help="Do not use CLS token", action="store_true"
  )
  parser.add_argument(
    '-C', '--chkpt_dir', help="Checkpoint directory", type=str, default="",
  )
  parser.add_argument(
    '--checkpoint', help="Checkpoint to continue from", type=str, default="",
  )
  parser.add_argument(
    '--seed', help="Experiment seed", type=int, default=1111,
  )
  parser.add_argument(
    '--inv_temp', help="Inverse temperature for softmax attn", type=float, default=None,
  )
  parser.add_argument(
    '--ngtk', help="# of global tokens", type=int, default=0,
  )
  mlp_activations = ['relu', 'gelu', 'mish']
  parser.add_argument(
    '--mlpa', help="Activation in MLP", choices=mlp_activations,
    default=mlp_activations[0]
  )
  parser.add_argument(
    "--adam", help="Use Adam optimizer", action="store_true"
  )
  

  args = parser.parse_args()
  assert os.path.exists(args.train)
  assert os.path.exists(args.val)
  assert args.test is None or os.path.exists(args.test)
  assert args.demb > 1
  assert args.dmlp >= args.demb
  assert args.nblocks > 1 or (args.nblocks == -1 and args.checkpoint != "")
  assert args.nheads >= 1
  assert 0 < args.dropout < 1.
  assert args.init_lr > 0.
  assert 0 < args.lr_decay_rate < 1.
  assert args.bsz >= 1
  assert args.nepochs > 1
  assert args.output_dir == "" or os.path.exists(args.output_dir)
  assert args.chkpt_dir == "" or os.path.exists(args.chkpt_dir)
  assert args.checkpoint == "" or os.path.exists(args.checkpoint)
  assert args.mask in [ mask_choices[i] for i in [NM] ] or args.mask_size >= 2, (
    f"\n Mask size should be greater than 1 (selected: {args.mask_size})"
    f"\n if mask not in type {[ mask_choices[i] for i in [NM] ]} "
    f"(selected: {args.mask})"
  )
  assert (
    (args.ngtk <= 0)
    or args.mask in [mask_choices[WM], mask_choices[BM]]
  ), (
    f"Number of global tokens can be > 0 (provided: {args.ngtk}) only "
    f"for masks [{mask_choices[WM]}, {mask_choices[BM]}] "
    f"(provided: {args.mask})"
  )
  assert args.inv_temp is None or args.inv_temp > 0.

  NOCLS = args.nocls
  MASK = args.mask
  MASKSIZE = args.mask_size
  INITLR = args.init_lr
  LRDRATE = args.lr_decay_rate
  SEED = args.seed
  INVTEMP = args.inv_temp
  NGTK = args.ngtk
  MLPA = args.mlpa

  checkpoint = None if args.checkpoint == "" else torch.load(args.checkpoint)
  new_nepochs = args.nepochs
  old_nepochs = 0
  if checkpoint is not None:
    print(f"Validating checkpoint {args.checkpoint} ....")
    old_args_dict = checkpoint['cli_args_dict']
    old_nepochs = checkpoint['epoch']
    assert old_args_dict['nepochs'] == old_nepochs
    assert new_nepochs > old_nepochs
    assert args.train == old_args_dict['train']

  # output file
  dname = args.train.split('/')[-2]
  ofile = dname
  cli_args = args.__dict__ if checkpoint is None else checkpoint['cli_args_dict']
  old_ofile = "" if checkpoint is None else checkpoint['train_stats']
  if cli_args['nepochs'] != new_nepochs:
    assert new_nepochs > cli_args['nepochs']
    print(
      f"Ignoring all CLI args except 'nepochs';"
      f" using rest from checkpoint {args.checkpoint}"
    )
    cli_args['nepochs'] = new_nepochs
    NOCLS = cli_args['nocls']
    MASK = cli_args['mask']
    MASKSIZE = cli_args['mask_size']
    INITLR = cli_args['init_lr']
    LRDRATE = cli_args['lr_decay_rate']
    SEED = cli_args['seed']
    INVTEMP = cli_args['inv_temp'] if 'inv_temp' in cli_args else None
    NGTK = cli_args['ngtk'] if 'ngtk' in cli_args else 0
    MLPA = cli_args['mlpa'] if 'mlpa' in cli_args else mlp_activations[0]

  for k, v in cli_args.items():
    if k in [
        "train", "val", "test",
        "output_dir", "chkpt_dir", "checkpoint"
    ]:
      continue
    if k == 'inv_temp' and v is None:
      continue
    if k == 'ngtk' and v == 0:
      continue
    if k == 'adam' and v == False:
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
    if old_ofile != "":
      print(f" ... will also be appending stats from {old_ofile}")
    assert old_ofile != ofile
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

  # read in data
  train_data = pd.read_csv(args.train, sep='\t')
  val_data = pd.read_csv(args.val, sep='\t')
  test_data = None if args.test is None else pd.read_csv(args.test, sep='\t')
  print(f"Train: {train_data.shape}, Val:{val_data.shape}")

  # Create token vocab
  seq_lens = [
    len(item)
    for item in yield_tokens(
      pd.concat([train_data, val_data]),
      cls_token=not NOCLS
    )
  ]
  MAXLEN = np.max(seq_lens)
  print(
    f"Sequence lengths ranging from {np.min(seq_lens)} to {MAXLEN}"
  )

  # adjust maxlen to match mask_size
  if MASKSIZE != -1:
    nblocks = (MAXLEN - int(not NOCLS)) // MASKSIZE
    if nblocks * MASKSIZE < (MAXLEN -  int(not NOCLS)):
      MAXLEN = int(not NOCLS) + (nblocks + 1) * MASKSIZE
    assert (MAXLEN - int(not NOCLS)) % MASKSIZE == 0
    print(f"Changing max length from {np.max(seq_lens)} -> {MAXLEN}")

  vocab = build_vocab_from_iterator(
    yield_tokens(
      pd.concat([train_data, val_data]),
      max_len=MAXLEN,
      cls_token=not NOCLS,
      ngtk=NGTK,
    )
  )
  print(
    f"Created vocab with {len(vocab)} tokens, with {CLS} token index:"
    f" {vocab([CLS]) if vocab.__contains__(CLS) else 'NA'}, \n"
    f"and with {PAD} token index: {vocab([PAD])}\n"
    f" (Max) sequence length: {MAXLEN}+{NGTK}"
  )
  print("Tokens:", vocab.lookup_tokens(range(len(vocab))))

  train_X = torch.tensor([
    vocab(item) for item in yield_tokens(
      train_data, max_len=MAXLEN, cls_token=not NOCLS, ngtk=NGTK,
    )
  ], dtype=torch.long)
  train_y = torch.tensor(train_data['Target'], dtype=torch.long)
  print('Train:', train_X.shape, train_y.shape)
  # print(type(train_X), type(train_y))

  val_X = torch.tensor([
    vocab(item) for item in yield_tokens(
      val_data, max_len=MAXLEN, cls_token=not NOCLS, ngtk=NGTK,
    )
  ], dtype=torch.long)
  val_y = torch.tensor(val_data['Target'], dtype=torch.long)
  print('Val:', val_X.shape, val_y.shape)

  test_X = None if test_data is None else torch.tensor([
    vocab(item) for item in yield_tokens(
      test_data, max_len=MAXLEN, cls_token=not NOCLS, ngtk=NGTK,
    )
  ], dtype=torch.long)
  test_y = None if test_data is None else torch.tensor(
    test_data['Target'], dtype=torch.long
  )
  if test_data is not None:
    print('Test:', test_X.shape, test_y.shape)


  NTOKEN = len(vocab)
  NCLASS = len(train_data["Target"].unique())

  print(
    f"Problem with {train_data.shape[0]} train and {val_data.shape[0]} val "
    f"examples with max length {MAXLEN}+{NGTK} "
    f"using a total of {NTOKEN} tokens and {NCLASS} classes"
  )

  model = TFClassifier(
    ntoken=NTOKEN,
    nclass=NCLASS,
    maxlen=MAXLEN+NGTK,
    d_emb=cli_args['demb'],
    d_qkv=cli_args['demb'],
    d_mlp=cli_args['dmlp'],
    nlayers=cli_args['nblocks'],
    nheads=cli_args['nheads'],
    dropout=cli_args['dropout'],
    mask_type=MASK,
    mask_size=MASKSIZE,
    cls_token=not NOCLS,
    inv_temp=1. if INVTEMP is None else INVTEMP,
    ngtk=NGTK,
    mlpa=MLPA,
  )

  criterion = nn.CrossEntropyLoss()
  lr = INITLR  # learning rate
  OPT = torch.optim.Adam if args.adam else torch.optim.SGD
  optimizer = OPT(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 1.0, gamma=LRDRATE
  )
  if checkpoint is not None:
    print(f"Loading states from checkpoint '{args.checkpoint}'")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  train_stats = []
  train_cols = ["epoch", "tr-ce", "tr-ac", "va-ce", "va-ac"]
  if test_data is not None:
    train_cols += ["te-ce", "te-ac"]
  current_best_val_acc = (
    0.0 if checkpoint is None else checkpoint['best_val_loss']
  )
  old_train_stats = None
  if old_ofile != "":
    assert checkpoint is not None
    old_train_stats = pd.read_csv(old_ofile)
    assert list(old_train_stats) == train_cols

  if GPUE:
    print("Moving model to GPU")
    model = model.to('cuda')
  print(
    f"Initializing epochs {old_nepochs}->{new_nepochs} with"
    f" current best validation accuracy {current_best_val_acc:.2f}..."
  )
  for epoch in range(old_nepochs + 1, new_nepochs + 1):
    epoch_start_time = time.time()
    epoch_stats = [epoch]
    train_epoch(
      model,
      train_X,
      train_y,
      NCLASS,
      criterion,
      optimizer,
      scheduler,
      cli_args['bsz'],
      RNG,
      epoch,
    )
    elapsed = time.time() - epoch_start_time
    teval = time.time()
    train_loss, train_acc = evaluate(
      model,
      train_X,
      train_y,
      NCLASS,
      criterion,
      cli_args['bsz'],
    )
    teval -= time.time()
    teval *= -1
    epoch_stats += [train_loss, train_acc]
    veval = time.time()
    val_loss, val_acc = evaluate(
      model,
      val_X,
      val_y,
      NCLASS,
      criterion,
      cli_args['bsz'],
    )
    veval -= time.time()
    veval *= -1
    epoch_stats += [val_loss, val_acc]
    print('-' * 89)
    if val_acc > current_best_val_acc:
      current_best_val_acc = val_acc
      # Saving current best model
      if args.chkpt_dir != "":
        torch.save(model.state_dict(), cfile_best)
    print(
      f'| epoch {epoch:3d} |'
      f' {elapsed:5.0f}s/{teval:5.0f}s/{veval:5.0f}s |'
      f' t-loss: {train_loss:.2f} | v-loss: {val_loss:.2f} |'
      f' t-acc: {train_acc:.2f} | v-acc: {val_acc:.2f} |'
    )
    scheduler.step()
    if test_data is not None:
      val_loss, val_acc = evaluate(
        model,
        test_X,
        test_y,
        NCLASS,
        criterion,
        cli_args['bsz'],
      )
      epoch_stats += [val_loss, val_acc]
    train_stats += [epoch_stats]
    print('-' * 89)
    if args.output_dir != "":
      print(f"Saving training stats in {ofile} ...")
      cur = pd.DataFrame(
        train_stats, columns=train_cols
      )
      if old_train_stats is not None:
        print(f" ... with old stats appended.")
        cur = pd.concat([old_train_stats, cur])
      cur.to_csv(ofile, header=True, index=False)
  # Saving last epoch model weights + relevant expt info
  if args.chkpt_dir != "":
    print(f"Saving checkpoint of final state in\n{cfile_last}\n")
    torch.save({
      'cli_args_dict': cli_args,
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
  if old_train_stats is not None:
    res_df = pd.concat([old_train_stats, res_df])
  print(res_df.tail(10))
  print(f"Completed with best validation accuracy: {current_best_val_acc:.2f}")
