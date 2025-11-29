#!/usr/bin/env python

import argparse
import os
import sys
from itertools import product
from pprint import pprint

import numpy as np
import pandas as pd
pd.set_option('display.precision', 2)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-O', '--output_dir', help="Output directory", type=str, required=True,
  )
  parser.add_argument(
    '--precision', help="mask size to use for compact figure", default=0.1, type=float,
  )
  parser.add_argument(
    '--xcomp', help="Do the xbound", action="store_true",
  )

  args = parser.parse_args()
  assert os.path.exists(args.output_dir)
  assert args.precision > 0.0

  delta_s = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0, 16.0, 32.0]
  dcolors = [
    'xkcd:periwinkle',
    'xkcd:magenta',
    'xkcd:black',
    'xkcd:purple',
    'xkcd:green',
    'xkcd:blue',
    'xkcd:brown',
    'xkcd:red',
    'xkcd:aquamarine',
    'xkcd:forest green',
    'xkcd:mauve',
  ]
  cdict = {d: c for d, c in zip(delta_s, dcolors)}
  c1_vals = np.arange(args.precision, 1, step=args.precision)
  c2_vals = np.arange(args.precision, 1, step=args.precision)

  def wratio(ds, c1, beta, c2s, mult=2):
    assert beta == -1
    dens = ds * c2s
    lvals = mult * (1.0 + (1.0 / dens))
    vals = c1 + ((1.0 / ds) * np.log(lvals))
    # print(f"RHS: 1, LHS in ({np.min(vals):.3f}, {np.max(vals):.3f})")
    return vals < 1

  def xratio(ds, c1, beta, c2s):
    assert beta >= 1
    # NOTE: Assuming 2 * gamma = delta_s
    dens = ds * c2s
    lvals = ds * (1 + beta) * (1.0 + (1.0 / dens))
    lvals += beta
    vals = c1 + ((1.0 / ds) * np.log(lvals))
    rhs = 1.0 + ((1 / ds) * np.log(ds + 1))
    # print(f"RHS: {rhs:.3f}, LHS in ({np.min(vals):.3f}, {np.max(vals):.3f})")
    return vals < rhs

  bndcomp = xratio if args.xcomp else wratio
  betas = [1, 2, 4, 8, 16, 32] if args.xcomp else [-1]
  cnames = ['delta-s', 'c1', 'c2', 'beta']
  res = []
  for beta in betas:
    for d in delta_s:
      print((
        f"Processing delta-s: {d}"
        + (f"(beta: {beta})" if args.xcomp else "")
        + " ..."
      ))
      for c in c1_vals:
        tvals = bndcomp(d, c, beta, c2_vals)
        if not (np.any(tvals)):
          continue
        minc2 = np.min(c2_vals[tvals])
        res += [(d, c, minc2, beta)]
  alldf = pd.DataFrame(res, columns=cnames)
  print(alldf.head(5))
  print(alldf.shape)
  fig, axs = plt.subplots(
    1, len(betas),
    figsize=(2.5*len(betas), 3.5) if args.xcomp else (3,3),
    sharex='row', sharey='row',
    squeeze=True,
  )
  if not args.xcomp:
    axs = [axs]
  axs[0].set_ylabel(r"$c_2$", fontsize=15)
  for beta, ax in zip(betas, axs):
    ax.set_xlabel(r"$c_1$", fontsize=15)
    for (d, b), dbdf in alldf.groupby(['delta-s', 'beta']):
      if b != beta:
        continue
      ax.plot(
        dbdf['c1'], dbdf['c2'],
        color=cdict[d],
        # 'black', linestyle='--',
        label=f"{d}",
      )
    ax.set_xlim(args.precision, 1)
    ax.set_ylim(args.precision, 1)
    ax.set_aspect('equal', adjustable='box')
    if args.xcomp:
      ax.set_title(r"$\beta=$" + f"{beta}", fontsize=15)
  axs[0].legend(
    loc='best',
    title=r'$\delta_s$' + (r"$=2\Gamma \Xi^2$" if args.xcomp else ""),
    fontsize=8,
    handlelength=1,
  )
  fig.suptitle(
    (r"$\lambda_X(\xi_s)$" + " vs " r"$\lambda_X(\xi_h)$")
    if args.xcomp else
    (r"$\lambda_W(\xi_s)$" + " vs " r"$\lambda_W(\xi_h)$"),
    fontsize=15,
  )
  fig.tight_layout()
  lfile = os.path.join(
    args.output_dir,
    ('xratio' if args.xcomp else 'wratio') + '.pdf'
  )
  print(f"Saving figures in {lfile} ...")
  fig.savefig(lfile)
