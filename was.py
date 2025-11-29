#!/usr/bin/env python

import argparse
import os
import math
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

  def set_weights(self, M, x, y):
    for (m, o, dx, dy) in zip(M.parameters(), self.origin, self.xdir, self.ydir):
      m.data = o + (x * dx) + (y * dy)
    return M

  def reset_weights(self, M):
    for (m, o) in zip(M.parameters(), self.origin):
      m.data = o
    return M


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
    '--pseed', help='Seed for picking projection directions',
    type=int, required=True,
  )
  parser.add_argument(
    '--aub', help='Max absolute range', type=float, default=1.0
  )
  parser.add_argument(
    '--nsteps', help='Number of values per axis', type=int, default=10
  )
  parser.add_argument(
    '--nbatches', help='Number of batches to process', type=int, default=1
  )

  args = parser.parse_args()
  assert os.path.exists(args.checkpoint)
  assert args.output == "" or os.path.exists(args.output)
  save_output = args.output != ""
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
      + f"__bsz2:{args.batch_size}__nb:{args.nbatches}"
      + f"__aub:{args.aub}__nsteps:{args.nsteps}__pseed:{args.pseed}.csv"
    )
    outfile = outfile.replace(outfile.split('__')[0], ddir)
    outfile = os.path.join(args.output, ddir, outfile)
    print(f"Saving the width-gap stats in {outfile}")
    assert not os.path.exists(outfile), f"Outfile {outfile} EXISTS --- QUITTING"
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

  pw = PerturbedWeights(orps, dir1, dir2)
  xyvals = torch.tensor([0]) if args.aub == 0 else torch.linspace(-args.aub, args.aub, steps=args.nsteps)
  print(f"Number of values per axis: {xyvals.size(0)}")

  astats = []
  anames = ['metric', 'lidx', 'sidx', 'qidx', 'dx', 'dy', 'val']
  mnames = [
    'V-norm-Upsilon',
    'QK-norm-Gamma',
    'P-norm-B',
    'R-norm-B',
    'width-delta',
    'sep-Delta',
    'beta'
  ]
  midxs = np.arange(len(mnames)).tolist()
  VN, QKN, PN, RN, WD, SD, BK = midxs

  def get_wb(module):
    return torch.hstack([module.weight, module.bias.unsqueeze(0).t()])

  for (xidx, dx) in enumerate(xyvals):
    for (yidx, dy) in enumerate(xyvals):
      model = pw.set_weights(model, dx, dy)
      model.eval()
      with torch.no_grad():
        # First compute weight norms
        for lidx, block in enumerate(model.encoder.transformer):
          # P-norm
          Pmat = get_wb(block.ffn[0])
          astats += [(
            mnames[PN], lidx, -1, -1, dx.item(), dy.item(),
            torch.linalg.matrix_norm(Pmat, ord=2).item()
          )]
          # R-norm
          Rmat = get_wb(block.ffn[3])
          astats += [(
            mnames[RN], lidx, -1, -1, dx.item(), dy.item(),
            torch.linalg.matrix_norm(Rmat, ord=2).item()
          )]
          # V-norm
          Vmat = get_wb(block.attention.W_v)
          astats += [(
            mnames[VN], lidx, -1, -1, dx.item(), dy.item(),
            torch.linalg.matrix_norm(Vmat, ord=2).item()
          )]
          # QK-norm
          Q = get_wb(block.attention.W_q)
          K = get_wb(block.attention.W_k)
          W = block.attention.inv_temp * (Q.t() @ K) / math.sqrt(model.d_model)
          astats += [(
            mnames[QKN], lidx, -1, -1, dx.item(), dy.item(),
            torch.linalg.matrix_norm(W, ord=2).item()
          )]
          print(f"Layer {lidx} weight norms computed")
        nbatches = 0
        for X, y in trdl:
          if GPUE:
            X = X.to('cuda')
          X = model.embedding(X) * math.sqrt(model.d_model)
          X = model.pos_encoder(X)
          for lidx, block in enumerate(model.encoder.transformer):
            # Emulate forward pass manually
            XX = block.norm_attn(X)
            # - extract dot-product matrix
            QQ = block.attention._reshape_qkv(block.attention.W_q(XX))
            KK = block.attention._reshape_qkv(block.attention.W_k(XX))
            DD = block.attention.inv_temp * (QQ @ KK.transpose(-2, -1)) / math.sqrt(KK.size(-1))
            DDwidths, DDseps, BB = None, None, None
            if MASKSIZE == -1:
              DDwidths = (
                torch.max(DD, dim=-1, keepdims=False).values
                - torch.min(DD, dim=-1, keepdims=False).values
              )
            else:
              assert block.attention.mask_attn is not None
              AA, _ = block.attention.mask_attn(DD, MASKSIZE, False, None, 0)
              MM = AA > 0
              ums = torch.unique(MM.sum(dim=-1))
              assert len(ums) == 1
              assert ums[0] == MASKSIZE
              BB = MM.sum(dim=-2, keepdims=False)
              UDD = DD[MM].reshape(DD.size(0), DD.size(1), DD.size(2), MASKSIZE)
              DDwidths = (
                torch.max(UDD, dim=-1, keepdims=False).values
                - torch.min(UDD, dim=-1, keepdims=False).values
              )
              MM = AA == 0
              MDD = DD[AA == 0].reshape(DD.size(0), DD.size(1), DD.size(2), DD.size(3)-MASKSIZE)
              DDseps = (
                torch.min(UDD, dim=-1, keepdims=False).values
                - torch.max(MDD, dim=-1, keepdims=False).values
              )
            assert DDwidths is not None
            # save the dot product widths
            for ii, widths in enumerate(DDwidths):
              assert widths.size(0) == 1
              astats += [
                (mnames[WD], lidx, ii, qq, dx.item(), dy.item(), wval.item())
                for qq, wval in enumerate(widths[0])
              ]
            if MASKSIZE != -1:
              assert (DDseps is not None) and (BB is not None)
              # save the per-query per-sample separation
              for ii, seps in enumerate(DDseps):
                assert seps.size(0) == 1
                astats += [
                  (mnames[SD], lidx, ii, qq, dx.item(), dy.item(), sval.item())
                  for qq, sval in enumerate(seps[0])
                ]
              # save the per-query per-sample unmasked key sum
              for ii, rsums in enumerate(BB):
                assert rsums.size(0) == 1
                astats += [
                  (mnames[BK], lidx, ii, qq, dx.item(), dy.item(), rval.item())
                  for qq, rval in enumerate(rsums[0])
                ]
            attn = block.attention(XX, XX, XX)
            XX = X + attn
            X = XX + block.ffn(block.norm_ffn(XX))
            print(f"[Batch {nbatches}] Layer {lidx} widths/separations/beta computed")
          # NOTE: Processing a single batch right now,
          #       the sample index counting assumes that
          nbatches += 1
          if nbatches >= args.nbatches:
            break
  astatsdf = pd.DataFrame(astats, columns=anames)
  print(f'All stats size: {astatsdf.shape}')

  fstats = []
  fnames = ['metric', 'lidx', 'q', 'dxdy', 'val']
  percs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
  thresholds = np.arange(0, 1, step=0.1)
  maxdxdy = astatsdf['dx'].max()
  assert maxdxdy == astatsdf['dy'].max()
  thresholds = thresholds[thresholds <= maxdxdy]
  print(f'Thresholds: {thresholds}')
  for (m,l), mdf in astatsdf.groupby(['metric', 'lidx']):
    for th in thresholds:
      print(f'Processing metric: {m} at level {l} ({mdf.shape}, threshold: {th:.2f})')
      vals = [
        row['val'] for idx, row in mdf.iterrows()
        if ((math.fabs(row['dx']) <= th) and (math.fabs(row['dy']) <= th))
      ]
      fstats += [(m, l, q, th, np.percentile(vals, q=q)) for q in percs]
  for m, mdf in astatsdf.groupby('metric'):
    for th in thresholds:
      print(f'Processing metric: {m} ({mdf.shape}, threshold: {th:.2f})')
      vals = [
        row['val'] for idx, row in mdf.iterrows()
        if ((row['dx'] <= th) and (row['dy'] <= th))
      ]
      fstats += [(m, -1, q, th, np.percentile(vals, q=q)) for q in percs]
  fstatsdf = pd.DataFrame(fstats, columns=fnames)
  print(f'Final stats size: {fstatsdf.shape}')
  print(fstatsdf[(fstatsdf['q'] == 95) & (fstatsdf['lidx'] == -1)].tail(60))
  print(fstatsdf[(fstatsdf['q'] == 5) & (fstatsdf['lidx'] == -1)].tail(60))
  if save_output:
    print(f"saving results in \n{outfile} ...")
    fstatsdf.to_csv(outfile, header=True, index=False)
