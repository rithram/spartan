#!/usr/bin/env python

import argparse
import os
import glob
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
    '-I', '--indir', help='Directory containing results', type=str, required=True,
  )
  parser.add_argument(
    '-R', '--regex', help='Expression to cover results', type=str, required=True,
  )
  parser.add_argument(
    '-O', '--output_dir', help="Output directory", type=str, required=True,
  )
  parser.add_argument(
    '-N', '--dname', help='Benchmark name', type=str, required=True,
  )
  metrics = ['ce', 'ac']
  parser.add_argument(
    '--metric', help='Metric to plot', choices=metrics, default=metrics[1],
  )
  parser.add_argument(
    '--include_extra', help="Include extra masks", action="store_true",
  )
  parser.add_argument(
    '--compact', help="single compact plot", action="store_true",
  )
  parser.add_argument(
    '--cmask_size', help="mask size to use for compact figure", default=-1, type=int,
  )
  splits = {'tr': "Training", 'va': "Held-out"}
  parser.add_argument(
    '--split', help='Split on which metric is computed',
    choices=splits.keys(), default=list(splits.keys())[0],
  )
  parser.add_argument(
    '--nolegend', help="Do not plot legend", action="store_true",
  )
  args = parser.parse_args()
  assert os.path.exists(args.indir)
  assert os.path.exists(args.output_dir)

  # output file
  oname = "agg__" + args.regex.replace("*", "__")
  ofile = oname.replace('.csv', '')
  OUTFILE = f"{args.dname}__{args.metric}__{ofile}.pdf"
  if args.include_extra:
    OUTFILE = OUTFILE.replace('.pdf', '__extra.pdf')
  print(f"Will be saving figure in {os.path.join(args.output_dir, OUTFILE)}")

  MASKS = ['none', 'windowed', 'blklocal', 'topk']
  if args.include_extra:
    MASKS += ['botk', 'rnd']
  MSIZES = [-1, args.cmask_size] if args.compact else [-1, 5, 9]
  NGTKS = [0, 1] if args.compact else [0, 1, 3]


  fnames = glob.glob(f"{args.indir}/{args.regex}")
  print(f"Found {len(fnames)} results files")
  print(fnames[0])
  configs = [f.split('/')[-1].replace('.csv', '').split("__") for f in fnames]
  maxopts = 0
  cnames = []
  for cnf in configs:
    tmp = [c.split(':')[0] for c in cnf]
    if len(tmp) > maxopts:
      maxopts = len(tmp)
      cnames = tmp
  print(f"Found a max of {maxopts} options:\n{cnames}")
  # verify the configurations
  cname_idxs = {c: i for (i, c) in enumerate(cnames)}
  new_configs = []
  for cnf in configs:
    map1 = {v.split(':')[0]: v.split(':')[1] for v in cnf[1:]}
    for c in cnames[1:]:
      if c not in map1:
        map1[c] = '0'
    tmp = [cnf[0]]
    for c in cnames[1:]:
      tmp += [f"{c}:{map1[c]}"]
    new_configs += [tmp]
  cnames[0] = 'data'
  # building the dataframe
  cdf = pd.DataFrame(new_configs, columns=cnames)
  if 'ngtk' not in cnames:
    cdf['ngtk'] = "ngtk:0"
  if 'nocls' not in cdf:
    cdf['nocls'] = 'nocls:True'
  cdf['fname'] = fnames
  print(f"Found a total of {cdf.shape} files")

  rlist = []
  for idx, row in cdf.iterrows():
    if row['mask'].split(':')[1] not in MASKS:
      continue
    if int(row['mask_size'].split(':')[1]) not in MSIZES:
      continue
    if int(row['ngtk'].split(':')[1]) not in NGTKS:
      continue
    rlist += [row]
  cdf = pd.DataFrame(rlist)
  print(f"Remaining files post-filtering {cdf.shape}")
  assert (
    len(cdf['nocls'].unique()) == 1
    and cdf['nocls'].unique()[0] == 'nocls:True'
  ), (
    f"unique vals: {cdf['nocls'].unique()}"
  )

  methods = []
  mstr = lambda m, s, g: (
    f"{m.split(':')[-1]}({s.split(':')[-1]})+G{g.split(':')[-1]}"
    .replace('+G0', '')
    .replace('-1','')
    .replace('()','')
    .replace('none', 'full')
    .replace('window', 'band')
  )

  for (m, s, g) in product(MASKS, MSIZES, NGTKS):
    if (m == 'none' and s != -1) or (s == -1 and m != 'none'):
      continue
    if m in ['none', 'topk', 'botk', 'rnd'] and g != 0:
      continue
    methods += [mstr(m, str(s), str(g))]
  print(f"{len(methods)} methods:\n{methods}")

  cdict = {
    'full': 'xkcd:almost black',
    'banded(5)': 'xkcd:brownish pink',
    'banded(9)': 'xkcd:peacock blue',
    'banded(5)+G1': 'xkcd:raspberry',
    'blklocal(9)': 'xkcd:salmon',
    'blklocal(5)': 'xkcd:blue',
    'blklocal(5)+G1': 'xkcd:olive green',
    'topk(9)': 'xkcd:dark orange',
    'topk(5)': 'xkcd:aquamarine',
    'botk(5)': 'xkcd:olive green',
    'botk(9)': 'xkcd:raspberry',
    'rnd(5)': 'xkcd:bright purple',
    'rnd(9)': 'xkcd:cadet blue',
  }
  nplotcols = 1 if args.compact else 2 * len(NGTKS) + 1
  if args.include_extra:
    nplotcols += 2
  nplotrows = 1
  statssets = ['tr', 'va']
  MDICT = {'ac': "Accuracy", 'ce': "Loss"}
  YLAB = f"{splits[args.split]} {MDICT[args.metric]}"
  fig, axs = plt.subplots(
    nplotrows, nplotcols, sharex=True, sharey=True,
    figsize=(2.5*nplotcols + 1, 2.5 * nplotrows)
  )
  if nplotcols == 1:
    axs = [axs]

  groups = cdf.groupby(['mask', 'mask_size', 'ngtk'])
  print(f"Found {groups.ngroups} groups of results")

  def plot_set(
      ax, df, label, color,
      pltcol=f'{args.split}-{args.metric}',
      repcol=f'va-{args.metric}',
      higher=args.metric=='ac'):
    print(f"Plotting {df.shape} lines in {color} for {label} ...")
    aggstats = []
    best_rep_val, best_rep_iter, last_rep_val, cvg_iter = [], [], [], []
    for idx, (i, r) in enumerate(df.iterrows()):
      print(f"--> rep {idx+1}/{df.shape[0]}")
      lab = mstr(r['mask'], r['mask_size'], r['ngtk'])
      assert lab == label, (f"Expected: {label}, got: {lab}")
      resdf = pd.read_csv(r['fname'])
      # saving stats to compute agg
      aggstats += [resdf[pltcol]]
      citer = np.nan
      citers = [i for i, v in enumerate(resdf[pltcol].values) if v >= 95]
      if len(citers) > 0:
        citer = np.min(citers)
      cvg_iter += [citer]
      best_rep_val += [resdf[repcol].max() if higher else resdf[repcol].min()]
      best_rep_iter += [np.argsort(resdf[repcol].values)[-1 if higher else 0]]
      last_rep_val += [resdf[repcol].values[-1]]
    # plotting agg stats
    retdict = {
      'bestval': {
        'mean': np.mean(best_rep_val), 'std': np.std(best_rep_val),
      },
      'bestiter': {
        'mean': np.mean(best_rep_iter), 'std': np.std(best_rep_iter),
      },
      'lastval': {
        'mean': np.mean(last_rep_val), 'std': np.std(last_rep_val),
      },
      'cvgiter': {
        'mean': np.mean(cvg_iter), 'std': np.std(cvg_iter),
      },
    }
    allstats = np.array(aggstats)
    median = np.percentile(allstats, q=50, axis=0)
    lqr = np.percentile(allstats, q=25, axis=0)
    uqr = np.percentile(allstats, q=75, axis=0)
    ax.plot(
      np.arange(median.shape[0]),
      median,
      color=color,
      linewidth=(1.25 if args.compact else 2.5),
      label=label.replace('banded', 'band').replace('blklocal', 'bloc').replace('+G', ':').upper(),
    )
    ax.fill_between(
      np.arange(median.shape[0]),
      lqr, uqr,
      color=color,
      alpha=(0.15 if args.compact else 0.2),
    )
    return retdict


  resdict = {}
  cidx = 0
  for m in MASKS:
    if m == 'none':
      continue
    for g in NGTKS:
      if m in ['topk', 'botk', 'rnd'] and g != 0:
        continue
      print(f"Col: {cidx+1} --> ({m}, {g})")
      for s in MSIZES:
        if s == -1:
          continue
        label = mstr(m, str(s), str(g))
        setdf = [gdf for ((mm, ss, gg), gdf) in groups if mstr(mm, ss, gg) == label]
        assert len(setdf) == 1, f"[mask:{m} #gtk:{g} msize:{s}] got: {len(setdf)} sets:\n {setdf}"
        print(f"---> plotting set {label} in column {cidx+1}")
        resdict[label] = plot_set(
          axs[cidx], setdf[0], label,
          cdict[label if args.compact else label.split('+')[0]]
        )
      splottitle = f"{m.replace('window', 'band').upper()}{f'+G{g}' if g > 0 else ''}"
      if not (args.compact):
        axs[cidx].set_title(splottitle, fontsize=12)
        cidx += 1
  for (grp, gdf) in groups:
    m, s, g = grp
    label = mstr(m, s, g)
    if m != 'mask:none':
      continue
    assert label == 'full', f"expected: full, got: {label}"
    print(f"Plotting {label} results on all plots")
    for ax in axs:
      resdict['full'] = plot_set(ax, gdf, label, cdict[label])
      ax.minorticks_on()
      ax.grid(which='major', axis='both', alpha=0.9)
      ax.grid(which='minor', axis='both', alpha=0.2)
    break

  if args.compact:
    axs[0].set_ylabel(YLAB, fontsize=10)
    axs[0].set_xlabel('Epochs', fontsize=10)
  assert cidx == len(axs) or (cidx == 0 and args.compact)
  for ax in axs:
    if not args.nolegend:
      ax.legend(
        ncol=1,
        fontsize=(7.5 if args.compact else 8),
        loc='best',
        handlelength=(0.75 if args.compact else 1),
      )

  fig.tight_layout()
  lfile = os.path.join(args.output_dir, OUTFILE)
  print(f"Saving figures in {lfile} ...")
  fig.savefig(lfile)

  allstats = []
  allcols = ['method', 'bval-m', 'bval-s', 'biter-m', 'biter-s', 'lval-m', 'lval-s', 'cvg-m', 'cvg-s']
  alltypes = [str, float, float, np.int32, np.int32, float, float, "Int64", "Int64"]
  alltypedict = {k: v for k, v in zip(allcols, alltypes)}
  for k, v in resdict.items():
    vals = []
    for kk, vv in v.items():
      vals += [vv['mean'], vv['std']]
    allstats += [(k, *vals)]
  alldf = pd.DataFrame(allstats, columns=allcols)
  for k in ['biter-m', 'biter-s', 'cvg-m', 'cvg-s']:
    print(f"converting type for {k} ...")
    alldf[k] = alldf[k].fillna(-1).astype(int)
  print(alldf)
