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
from torch.utils.data import DataLoader

print(f"Cuda available: {torch.cuda.is_available()}")
GPUE = torch.cuda.is_available()

from tf_model import TFClassifier
from datasets import ParityCheck, UnequalPairs, MissingDuplicate
from datasets import CycNav, MAB, SolEq, StackMan

def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    nclass: int,
    crit,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    log_interval: int = 25,
) -> None:
  model.train()  # turn on train mode
  total_loss = 0.
  start_time = time.time()

  num_batches = len(data_loader)
  for batch, (X, y) in enumerate(data_loader):
    data = X.to('cuda') if GPUE else X
    targets = y.to('cuda') if GPUE else y
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
    data_loader: DataLoader,
    nclass: int,
    crit,
) -> (float, float):
  model.eval()  # turn on evaluation mode
  total_loss = 0.
  total_correct = 0
  den = 0
  with torch.no_grad():
    for X, y in data_loader:
      den += X.size(0)
      if GPUE:
        X = X.to('cuda')
        y = y.to('cuda')
      output = model(X)
      output_flat = output.view(-1, nclass)
      total_loss += (X.size(0) * crit(output_flat, y).item())
      _, preds = torch.max(output, 1)
      total_correct += (y == preds).sum().item()
  return (
    total_loss / den,
    total_correct * 100.0 / den
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-T', '--ntrain', help='# training data', type=int, required=True,
  )
  parser.add_argument(
    '-V', '--nval', help='# validation data', type=int, required=True,
  )
  parser.add_argument(
    '-t', '--ntest', help='# test data', type=int, default=0,
  )
  parser.add_argument(
    '--min_len', help='Minimum sequence length', type=int, required=True,
  )
  parser.add_argument(
    '--max_len', help='Maximum sequence length', type=int, required=True,
  )
  parser.add_argument(
    '--len_step', help='Steps to increase length', type=int, required=True,
  )
  data_choices = {
    'parity': ParityCheck,
    'ueqpairs': UnequalPairs,
    'missdup': MissingDuplicate,
    'cycnav': CycNav,
    'mab': MAB,
    'soleq': SolEq,
    'stackman': StackMan,
  }
  parser.add_argument(
    '--data', help='Data choice',
    choices=list(data_choices.keys()), required=True,
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
    '-M', '--mask', help='Type of attention masks',
    choices=mask_choices, default=mask_choices[NM],
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


  args = parser.parse_args()
  assert args.ntrain > 10 and args.nval > 10
  assert args.ntest == 0 or args.ntest > 10
  assert 10 < args.min_len <= args.max_len
  assert args.len_step >= 0
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
  assert args.mask == mask_choices[NM] or args.mask_size >= 2, (
    f"\n Mask size should be greater than 1 (selected: {args.mask_size})"
    f"\n if mask not of type {mask_choices[NM]} (selected: {args.mask})"
  )
  assert args.inv_temp is None or args.inv_temp > 0.
  assert (
    (args.ngtk <= 0)
    or args.mask in [mask_choices[WM], mask_choices[BM]]
  ), (
    f"Number of global tokens can be > 0 (provided: {args.ngtk}) only "
    f"for masks [{mask_choices[WM]}, {mask_choices[BM]}] "
    f"(provided: {args.mask})"
  )

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
    assert args.data == old_args_dict['data']
    assert args.ntrain == old_args_dict['ntrain']
    assert args.min_len == old_args_dict['min_len']
    assert args.max_len == old_args_dict['max_len']
    assert args.len_step == old_args_dict['len_step']

  # output file
  dname = (
    f"{args.data}-l{args.min_len}-s{args.len_step}-L{args.max_len}"
    f"-{args.ntrain}-{args.nval}-{args.ntest}"
  )
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
        "data", "ntrain", "nval", "ntest",
        "min_len", "len_step", "max_len",
        "output_dir", "chkpt_dir", "checkpoint"
    ]:
      continue
    if k == 'inv_temp' and v is None:
      continue
    if k == 'ngtk' and v == 0:
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

  # Generate data
  Data = data_choices[args.data]
  train_data = Data(
    min_length=args.min_len,
    max_length=args.max_len,
    len_step=args.len_step,
    nsamples=args.ntrain,
    mask_size=MASKSIZE,
    ngtk=NGTK,
    seed=SEED,
    split="train",
  )
  trdl = DataLoader(train_data, batch_size=cli_args['bsz'], shuffle=True)
  NTOKEN = train_data.NTOKENS
  NCLASS = train_data.NLABELS
  val_data = Data(
    min_length=args.min_len,
    max_length=args.max_len,
    len_step=args.len_step,
    nsamples=args.nval,
    mask_size=MASKSIZE,
    ngtk=NGTK,
    seed=SEED,
    split="val",
  )
  vadl = DataLoader(val_data, batch_size=cli_args['bsz'], shuffle=False)
  assert val_data.NTOKENS == NTOKEN
  assert val_data.NLABELS == NCLASS
  assert val_data.xlen == train_data.xlen
  test_data = None
  tedl = None
  if args.ntest > 0:
    test_data = Data(
      min_length=args.min_len,
      max_length=args.max_len,
      len_step=args.len_step,
      nsamples=args.ntest,
      mask_size=MASKSIZE,
      ngtk=NGTK,
      seed=SEED,
      split="test",
    )
    assert test_data.NTOKENS == NTOKEN
    assert test_data.xlen == train_data.xlen
    tedl = DataLoader(test_data, batch_size=cli_args['bsz'], shuffle=False)

  print(
    f"Problem with {len(train_data)} train and {len(val_data)} val "
    f"examples with max length {train_data.xlen} "
    f"({train_data.max_length_padded}+{NGTK}) "
    f"using a total of {NTOKEN} tokens and {NCLASS} classes"
  )

  model = TFClassifier(
    ntoken=NTOKEN,
    nclass=NCLASS,
    maxlen=train_data.xlen,
    d_emb=cli_args['demb'],
    d_qkv=cli_args['demb'],
    d_mlp=cli_args['dmlp'],
    nlayers=cli_args['nblocks'],
    nheads=cli_args['nheads'],
    dropout=cli_args['dropout'],
    mask_type=MASK,
    mask_size=MASKSIZE,
    cls_token=False,
    inv_temp=1. if INVTEMP is None else INVTEMP,
    ngtk=NGTK,
    mlpa=MLPA,
  )

  criterion = nn.CrossEntropyLoss()
  lr = INITLR  # learning rate
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
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
      trdl,
      NCLASS,
      criterion,
      optimizer,
      scheduler,
      epoch,
    )
    elapsed = time.time() - epoch_start_time
    teval = time.time()
    train_loss, train_acc = evaluate(
      model,
      trdl,
      NCLASS,
      criterion,
    )
    teval -= time.time()
    teval *= -1
    epoch_stats += [train_loss, train_acc]
    veval = time.time()
    val_loss, val_acc = evaluate(
      model,
      vadl,
      NCLASS,
      criterion,
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
      assert tedl is not None
      test_loss, test_acc = evaluate(
        model,
        tedl,
        NCLASS,
        criterion,
      )
      epoch_stats += [test_loss, test_acc]
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
    # break
  # Saving last epoch model weights + relevant expt info
  if args.chkpt_dir != "":
    print(f"Saving checkpoint of final state in\n{cfile_last}\n")
    torch.save({
      'cli_args_dict': cli_args,
      'train_data': train_data,
      'val_data': val_data,
      'test_data': test_data,
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
