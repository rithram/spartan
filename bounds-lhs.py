#!/usr/bin/env python

import argparse
import glob
import os
import sys
from itertools import product
from pprint import pprint

import numpy as np
import pandas as pd

def wratio(ds, c1, beta, c2s, mult=2):
  assert beta == -1
  dens = ds * c2s
  lvals = mult * (1.0 + (1.0 / dens))
  vals = c1 + ((1.0 / ds) * np.log(lvals))
  return vals

def xratio(ds, c1, beta, c2s):
  assert beta >= 0.
  # NOTE: Assuming 2 * gamma * \xi^2 = delta_s
  dens = ds * c2s
  lvals = ds * (1 + beta) * (1.0 + (1.0 / dens))
  lvals += beta
  vals = c1 + ((1.0 / ds) * np.log(lvals)) - ((1 / ds) * np.log(ds + 1))
  return vals


NNZ = 5.0

## Results in the paper for reference:
## ListOps = [
##   [8.61, 18.5, 29.4, 87.8],
##   [3.51, 6.74, 9.67, 28.2],
##   [0.016, 0.005, 0.002, 1e-9],
##   [1, 3, 15, 598],
## ]
## Parity = [
##   [8.30, 10.1, 11.2, 19.4],
##   [2.31, 3.13, 3.78, 9.16],
##   [0.062, 0.022, 0.011, 1e-6],
##   [8, 13, 16, 33],
## ]
## EvenPairs = [
##   [2.03, 4.73, 9.44, 14.6],
##   [1.03, 2.84, 5.50, 8.25],
##   [0.009, 0.003, 0.002, 3e-8],
##   [6, 17, 26, 40],
## ]
## MissingDup = [
##   [4.63, 9.25, 17.1, 23.9],
##   [2.36, 4.25, 4.88, 10.5],
##   [0.018, 0.006, 0.003, 1e-7],
##   [7, 15, 21, 40],
## ]

def print_bound_lhs():
  for D in [ListOps, Parity, EvenPairs, MissingDup]:
    vals = np.array(D)
    print(D)
    for i in range(vals.shape[1]):
      col = vals[:, i]
      ds = col[0]
      c1 = col[1] / ds
      c2 = col[2] / ds
      print(f'- W: {wratio(ds, c1, -1, c2):.4f}')
      beta = col[3]
      print(f'- X: {xratio(ds, c1, beta/NNZ, c2):.4f}')


def get_ratios(fd, hhd, hhs, hhb):
  nstats = fd.shape[0]
  wr = np.array([
    wratio(fd[i], hhd[i]/fd[i], -1, hhs[i]/fd[i])
    for i in range(nstats)
  ])
  xr = np.array([
    xratio(fd[i], hhd[i]/fd[i], hhb[i]/NNZ, hhs[i]/fd[i])
    for i in range(nstats)
  ])
  return wr, xr

def get_stats(valdf, quantiles, reverse=False):
  return np.array([
    max(valdf[valdf['q'] == (100-q if reverse else q)]['val'].values[0], 1e-9)
    for q in quantiles
  ])
      
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--indir', help="Directory containing statistics for the bounds",
    required=True
  )
  args = parser.parse_args()
  assert os.path.isdir(args.indir)
  quantiles = np.array([75, 90, 95, 100])
  tasks = ['listops', 'parity', 'ueqpairs', 'missdup']
  for t in tasks:
    assert os.path.isdir(os.path.join(args.indir, t)), (
      f"Cannot find directory: {os.path.join(args.indir, t)}"
    )
    fnames = glob.glob(os.path.join(args.indir, t, "*.csv"))
    print(f'============ {t} ============')
    print(f"Found {len(fnames)} files for task {t}")
    print('-------------------------------')
    stfile = [f for f in fnames if "mask:none" in f]
    assert len(stfile) == 1
    tkfile = [f for f in fnames if "mask:topk__mask_size:5" in f]
    assert len(tkfile) == 1
    stdf = pd.read_csv(stfile[0])
    full_dis = stdf[(stdf['lidx'] == -1) & (stdf['metric'] == 'width-delta')]
    fdstats = get_stats(full_dis, quantiles)
    print(f'Full dispersion:\t{fdstats}')
    tkdf = pd.read_csv(tkfile[0])
    hh_dis = tkdf[(tkdf['lidx'] == -1) & (tkdf['metric'] == 'width-delta')]
    hh_sep = tkdf[(tkdf['lidx'] == -1) & (tkdf['metric'] == 'sep-Delta')]
    hh_beta = tkdf[(tkdf['lidx'] == -1) & (tkdf['metric'] == 'beta')]
    hdstats = get_stats(hh_dis, quantiles)
    hsstats = get_stats(hh_sep, quantiles, reverse=True)
    hbstats = get_stats(hh_beta, quantiles)
    print(f'HH dispersion:\t{hdstats}')
    print(f'HH separation:\t{hsstats}')
    print(f'HH sink ratio:\t{hbstats/NNZ}')
    wr, xr = get_ratios(fdstats, hdstats, hsstats, hbstats)
    print('-------------------------------')
    print(f'LHS(304):\t{wr}')
    print(f'LHS(305):\t{xr}')
    print('-------------------------------')
