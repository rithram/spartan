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
    '--nlevels', help='Number of loss levels', type=int, default=20,
  )
  parser.add_argument(
    '--masks', help="Masks to consider", type=str, required=True,
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
  assert args.nlevels > 3
  fname_prefix = 'lsurf' + args.lldata.split('/')[-1].replace(
    '__lsurf.csv', ''
  ).replace('*', '_').replace('__', '_')
  fcontour = fname_prefix + f'contour.pdf'
  fheatmap = fname_prefix + 'heatmap.pdf'
  fcpath = os.path.join(args.output, fcontour)
  fhpath = os.path.join(args.output, fheatmap)
  print(f"Saving figures in \n{fcpath}\n{fhpath}")

  lmap = {
    'none': 'STANDARD',
    'topk': 'TOPK',
    'windowed': 'BANDED',
    'blklocal': 'BLKLOCAL',
  }
  
  nrows = len(kfiles)
  cfig, caxs = plt.subplots(nrows, 1, figsize=(4, nrows*4), squeeze=True)
  hfig, haxs = plt.subplots(nrows, 1, figsize=(4, nrows*4), squeeze=True)

  for ridx, (kf, mask) in enumerate(zip(kfiles, mnames)):
    lldata = pd.read_csv(kf)
    print(lldata.shape)
    print(list(lldata))
    lldata['cerror'] = 100.0 - lldata['acc'].values
    print(list(lldata))
    xdata = np.sort(lldata['xval'].unique())
    ydata = np.sort(lldata['yval'].unique())
    print(xdata.shape, ydata.shape)
    nvals = len(xdata)
    assert nvals == len(ydata)
    lgrid = np.empty([nvals, nvals])
    xmin, xmax = xdata.min(), xdata.max()
    ymin, ymax = ydata.min(), ydata.max()
    xgap = xdata[1] - xdata[0]
    ygap = ydata[1] - ydata[0]
    xidx = ((lldata['xval'].values - xmin) / xgap).astype(int)
    print(xidx.min(), xidx.max(), xidx.shape)
    yidx = ((lldata['yval'].values - ymin) / ygap).astype(int)
    print(yidx.min(), yidx.max(), yidx.shape)
    m = 'loss'
    for (xx, yy, vv) in zip(xidx, yidx, lldata[m]):
      lgrid[xx, yy] = vv
    lmin, lmax = lldata[m].min()+1e-8, lldata[m].max()
    levels = np.exp(
      np.linspace(np.log(lmin), np.log(lmax), num=args.nlevels)
    )
    print(f"Levels: {levels}")
    # Create Contour Plot
    print("Creating and plotting contours ...")
    ax = caxs[ridx]
    ax.set_title(f"{lmap[mask]}")
    ax.set_aspect("equal")
    CS = ax.contour(
      xdata, ydata, lgrid,
      levels=levels, cmap="magma", linewidths=0.75,
      norm=mcols.LogNorm(vmin=lmin, vmax=lmax * 2.0),
    )
    ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
    # Create heatmap
    print("Plotting heatmap ...")
    ax = haxs[ridx]
    ax.set_title(f"{lmap[mask]}")
    ax.set_aspect("equal")
    ax.imshow(lgrid, cmap='inferno')

  for f, fp in zip([cfig, hfig], [fcpath, fhpath]):
    f.tight_layout()
    print(f"Saving figure in {fp} ...")
    f.savefig(fp)
