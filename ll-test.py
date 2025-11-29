#!/usr/bin/env python

import argparse
import os
import time
import sys
import random
from itertools import product
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

print(f"Cuda available: {torch.cuda.is_available()}")
GPUE = torch.cuda.is_available()

from tf_model import TFClassifier
from datasets import (
  ListOpsDataset,
  ParityCheck,
  UnequalPairs,
  MissingDuplicate,
  CycNav,
  MAB,
  SolEq,
  StackMan,
)


class PerturbedWeights(object):
  def __init__(self, origin, xdir, ydir):
    self.origin = origin
    self.xdir = xdir
    self.ydir = ydir

  def set_weights(self, M, x, y, yonly=False):
    for (m, o, dx, dy) in zip(M.parameters(), self.origin, self.xdir, self.ydir):
      m.data = o + (x * dx) + (y * dy)
    return M

  def reset_weights(self, M):
    for (m, o) in zip(M.parameters(), self.origin):
      m.data = o
    return M


def write_to_file(lsurf, cols, ofile, b1):
  if b1:
    pd.DataFrame(lsurf, columns=cols).to_csv(
      ofile, header=True, index=False
    )
  else:
    pd.DataFrame(lsurf, columns=cols).to_csv(
      ofile, mode='a', header=False, index=False
    )



NNCH = {
  'parity': ParityCheck,
  'ueqpairs': UnequalPairs,
  'missdup': MissingDuplicate,
  'cycnav': CycNav,
  'mab': MAB,
  'soleq': SolEq,
  'stackman': StackMan,
}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--checkpoint',
    help="Checkpoint to continue from",
    type=str,
    default="",
  )
  parser.add_argument(
    '--output', help="Directory for output", type=str, default="",
  )
  parser.add_argument(
    '--batch_size', help='Batch size', type=int, default=-1
  )
  parser.add_argument(
    '--aub', help='Max absolute range', type=float, default=1.0
  )
  parser.add_argument(
    '--nsteps', help='Number of values per axis', type=int, default=10
  )
  parser.add_argument(
    '--pseed', help='Seed for picking projection directions',
    type=int, required=True,
  )
  parser.add_argument(
    '--wbsz', help='Batch size for appending',
    type=int, default=100,
  )

  args = parser.parse_args()
  assert os.path.exists(args.checkpoint)
  assert args.output == "" or os.path.exists(args.output)
  save_output = args.output != ""
  assert args.aub > 0.0
  assert args.nsteps > 2
  checkpoint = torch.load(args.checkpoint)
  print(f"Validating checkpoint {args.checkpoint} ....")
  cli_args = checkpoint['cli_args_dict']
  BSZ = args.batch_size if args.batch_size != -1 else cli_args['bsz']

  print(cli_args)

  NOCLS = cli_args['nocls'] if 'nocls' in cli_args else True
  MASK = cli_args['mask']
  MASKSIZE = cli_args['mask_size']
  SEED = cli_args['seed']
  INVTEMP = cli_args['inv_temp'] if 'inv_temp' in cli_args else None
  NGTK = cli_args['ngtk'] if 'ngtk' in cli_args else 0
  MLPA = cli_args['mlpa'] if 'mlpa' in cli_args else 'relu'
  assert 'train' in cli_args or 'data' in cli_args
  TRAIN = cli_args['train'] if 'train' in cli_args else cli_args['data']
  VAL = cli_args['val'] if 'val' in cli_args else cli_args['data']
  print(TRAIN, VAL)
  assert TRAIN in NNCH.keys() or os.path.exists(TRAIN)
  assert VAL in NNCH.keys() or os.path.exists(VAL)
  nnch_data = TRAIN in NNCH.keys()
  ldf_prev = None
  if save_output:
    ddir = TRAIN
    if nnch_data:
      assert os.path.exists(os.path.join(args.output, TRAIN))
    else:
      assert os.path.exists(os.path.join(args.output, "listops"))
      ddir = "listops"

    outfile = (
      args.checkpoint.split('/')[-1].replace('_last.pt', '')
      + f"__aub:{args.aub}__nsteps:{args.nsteps}__pseed:{args.pseed}__"
      + "lsurf.csv"
    )
    outfile = os.path.join(args.output, ddir, outfile)
    print(f"Saving the loss surface in {outfile}")
    if os.path.exists(outfile):
      ldf_prev = pd.read_csv(outfile)
      print(f"Found existing loss surface:\n{outfile}")
      print(f"Surface size: {ldf_prev.shape}")
      if (args.nsteps * args.nsteps) == ldf_prev.shape[0]:
        print("Loss surface computation seems complete -- EXITING")
        sys.exit(0)
  # Set seeds for data generation for NNCH
  RNG = np.random.RandomState(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  # create dataloader for training data
  train_data = NNCH[TRAIN](
    min_length=cli_args["min_len"],
    max_length=cli_args["max_len"],
    len_step=cli_args["len_step"],
    nsamples=cli_args["ntrain"],
    mask_size=MASKSIZE,
    ngtk=NGTK,
    seed=SEED,
    split="train",
  ) if nnch_data else ListOpsDataset(
    {"train": TRAIN, "val": VAL},
    mask_size=MASKSIZE,
    ngtk=NGTK,
    split="train",
  )
  trdl = DataLoader(train_data, batch_size=BSZ, shuffle=False)
  NTOKENS = train_data.NTOKENS
  NCLASSES = len(train_data.y.unique())
  MAXLEN = train_data.X.shape[1]


  print(
    f"Problem with {len(train_data)} training samples with "
    f"maximum length {MAXLEN} (full shape: {train_data.X.shape}) "
    f"using a total of {NTOKENS} tokens and {NCLASSES} classes"
  )

  model = TFClassifier(
    ntoken=NTOKENS,
    nclass=NCLASSES,
    maxlen=MAXLEN,
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
  ).to('cuda')
  print(f"initializing the model ...")
  print(f"Loading states from checkpoint '{args.checkpoint}'")
  model.load_state_dict(checkpoint['model_state_dict'])

  # Set seeds for weight perturbation directions
  RNG = np.random.RandomState(args.pseed)
  torch.manual_seed(args.pseed)
  np.random.seed(args.pseed)
  random.seed(args.pseed)

  orps = []
  dir1 = []
  dir2 = []
  for p in model.parameters():
    with torch.no_grad():
      num = torch.linalg.norm(p).item()
      rnd1 = torch.randn_like(p)
      den1 = torch.linalg.norm(rnd1).item()
      rnd1 *= (num / den1)
      rnd2 = torch.randn_like(p)
      den2 = torch.linalg.norm(rnd2).item()
      rnd2 *= (num / den2)
      orps += [p.clone()]
      dir1 += [rnd1]
      dir2 += [rnd2]
  assert len(orps) == len(dir1) == len(dir2)

  criterion = nn.CrossEntropyLoss()
  from ch import evaluate
  start_time = time.time()
  train_loss, train_acc = evaluate(
    model,
    trdl,
    NCLASSES,
    criterion,
  )
  teval = time.time() - start_time
  print(
    f"Original metrics -- loss:{train_loss:.4f}, "
    f"acc:{train_acc:.4f}, time:{teval:.4f}"
  )

  pw = PerturbedWeights(orps, dir1, dir2)
  xyvals = torch.linspace(-args.aub, args.aub, steps=args.nsteps)
  print(f"Number of values per axis: {xyvals.size(0)}")
  loss_surface = []
  lcols = ['idx', 'xval', 'yval', 'loss', 'acc', 'time']
  done_idx = 0 if ldf_prev is None else ldf_prev['idx'].values.astype(np.int32)[-1]
  print(f"skipping first {done_idx} grid points ...")
  first_batch = True if done_idx == 0 else False
  eidx = 0
  for (xidx, dx) in tqdm(enumerate(xyvals)):
    for (yidx, dy) in enumerate(xyvals):
      eidx += 1
      if eidx <= done_idx:
        continue
      model = pw.set_weights(model, dx, dy)
      yonly = True
      start_time = time.time()
      train_loss, train_acc = evaluate(
        model, trdl, NCLASSES, criterion,
      )
      teval = time.time() - start_time
      loss_surface += [(eidx, dx.item(), dy.item(), train_loss, train_acc, teval)]
      if save_output and len(loss_surface) == args.wbsz:
        # batched appending instead of repeated writing
        write_to_file(loss_surface, lcols, outfile, first_batch)
        first_batch = False
        loss_surface = []
      model = pw.reset_weights(model)
  if save_output and len(loss_surface) > 0:
    # appending/writing the final batch
    write_to_file(loss_surface, lcols, outfile, first_batch)
    first_batch = False
  ldf = pd.read_csv(outfile) if save_output else pd.DataFrame(loss_surface, columns=lcols)
  assert ldf.shape[0] == (args.nsteps * args.nsteps)
  print(f"Loss surface size: {ldf.shape}")
  start_time = time.time()
  train_loss, train_acc = evaluate(
    model,
    trdl,
    NCLASSES,
    criterion,
  )
  teval = time.time() - start_time
  print(
    f"Final check metrics -- loss:{train_loss:.4f}, "
    f"acc:{train_acc:.4f}, time:{teval:.4f}"
  )
