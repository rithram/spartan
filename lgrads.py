#!/usr/bin/env python

import argparse
import glob
import os
import time
import sys

import numpy as np
import pandas as pd
import matplotlib.colors as mcols
import matplotlib.pyplot as plt


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--lldata', help="Loss surface data", type=str, required=True,
  )
  parser.add_argument(
    '--output', help="Directory for output", type=str, default="",
  )
  parser.add_argument(
    '--tstep', help='Step for radius', type=float, default=0.1
  )
  parser.add_argument(
    '--masks', help="Masks to consider", type=str, required=True,
  )
  parser.add_argument(
    '--uperc', help='Upper percentile', type=int, default=90,
  )
  parser.add_argument(
    '--xmax', help='Max perturbation from center', type=float, default=1.0,
  )
  args = parser.parse_args()
  fnames = glob.glob(args.lldata)
  print(f"Found data for {len(fnames)} loss surfaces ...")
  assert len(fnames) > 0
  masks = args.masks.split(':')
  print(f"Accepting {len(masks)} for now ...")
  assert len(masks) > 0
  kfiles = [f for f in fnames if any([f'mask:{m}' in f for m in masks])]
  print(f"Keeping around {len(kfiles)} surfaces ...")
  mnames = [k.split('__')[5].split(':')[1] for k in kfiles]
  print(mnames)
  assert len(kfiles) > 0
  assert os.path.exists(args.output)
  fname_prefix = 'lgrad_curves_' + args.lldata.split('/')[-1].replace(
    '__lsurf.csv', ''
  ).replace('*', '_').replace('__', '_')
  fboxfile = os.path.join(args.output, fname_prefix + f'.pdf')
  assert args.tstep > 0.0
  assert args.uperc > 50 and args.uperc < 100

  upercs =  [up for up in [50, 75, 95, 99] if up <= args.uperc]
  lstyles = [':', '-.', '--', '-'][-len(upercs):]

  print(f"Saving figures in \n{fboxfile}")

  lmap = {
    'none': 'FULL',
    'topk': 'TOPK',
    'windowed': 'BANDED',
    'blklocal': 'BLKLOCAL',
  }
  cmap = {
    'none': 'black',
    'topk': 'xkcd:aquamarine'
  }

  nrows = len(kfiles)
  print(f"Found {nrows} files ...")
  fig, ax = plt.subplots(1, 1, figsize=(4.5,4), squeeze=True)

  for ridx, (kf, mask) in enumerate(zip(kfiles, mnames)):
    print(f"reading {kf} ...")
    lldata = pd.read_csv(kf)
    xdata = np.sort(lldata['xval'].unique())
    ydata = np.sort(lldata['yval'].unique())
    print(xdata.shape, ydata.shape)
    nvals = len(xdata)
    assert nvals == len(ydata)
    lgrid = np.zeros([nvals, nvals])
    xmin, xmax = xdata.min(), xdata.max()
    ymin, ymax = ydata.min(), ydata.max()
    xgap = xdata[1] - xdata[0]
    ygap = ydata[1] - ydata[0]
    print(xgap, ygap)
    xidx = ((lldata['xval'].values - xmin) / xgap).astype(int)
    print(xidx.min(), xidx.max(), xidx.shape)
    yidx = ((lldata['yval'].values - ymin) / ygap).astype(int)
    print(yidx.min(), yidx.max(), yidx.shape)
    m = 'loss'
    for (xx, yy, vv) in zip(xidx, yidx, lldata[m]):
      lgrid[xx, yy] = vv
    thresholds = np.arange(args.tstep, 1, step=args.tstep)
    print(thresholds[thresholds < args.xmax])
    ubs = [[] for up in upercs]
    for thres in thresholds[thresholds < args.xmax]:
      xlb = int((-thres - xmin) / xgap)
      xub = int((thres - xmin) / xgap)
      ylb = int((-thres - ymin) / ygap)
      yub = int((thres - ymin) / ygap)
      tgrid = lgrid[xlb:xub, ylb:yub]
      print("Computing gradients ...")
      diff1 = np.abs(tgrid[1:,:] - tgrid[:-1,:]).reshape(-1)
      diff2 = np.abs(tgrid[:,1:] - tgrid[:,:-1]).reshape(-1)
      diff = np.hstack([diff1, diff2])
      print(
        f"[{xlb}-{xub} x {ylb}-{yub} ({diff.shape})] "
        f"{np.percentile(diff, q=args.uperc):.5f}, "
        f"{np.percentile(diff, q=50):.5f}, "
        f"{np.percentile(diff, q=100-args.uperc):.5f} "
        f"[{tgrid.min():.5f}, {tgrid.max():.5f}]"
      )
      for i, up in enumerate(upercs):
        ubs[i] += [np.percentile(diff, q=up) / xgap]
    for i, up in enumerate(upercs):
      ax.plot(
        thresholds[thresholds < args.xmax],
        np.array(ubs[i]),
        lstyles[i],
        color=cmap[mask],
        label=f"{lmap[mask]}:{up}%"
      )
    ax.legend(loc='best', ncol=1, handlelength=2, fontsize=8)
    ax.set_xlabel('Grid range around ' + r'$\Theta$', fontsize=12)
    ax.set_ylabel(
      'Distribution of '
      + r'$\frac{|\mathcal{L}(\Theta) - \mathcal{L}(\bar{\Theta}) |}{|| \Theta - \bar{\Theta} ||}$',
      fontsize=12
    )
  fig.tight_layout()
  print(f"Saving figure in {fboxfile} ...")
  fig.savefig(fboxfile)
