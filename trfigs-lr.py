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
    "--sharey", help="The plots have shared y-axis", action="store_true"
  )

  args = parser.parse_args()
  assert os.path.exists(args.indir)
  assert os.path.exists(args.output_dir)

  # output file
  oname = "agg__" + args.regex.replace("*", "__")
  ofile = oname.replace('.csv', '')
  OUTFILE = f"{args.dname}__{args.metric}__{ofile}.pdf"
  print(f"Will be saving figure in {os.path.join(args.output_dir, OUTFILE)}")

  MASKS = ['none', 'topk']
  MSIZES = [-1, 5]
  LRS = [1e-05, 0.0001, 0.001, 0.01, 0.1, 0.3, 1.0, 0.66, 1.5, 2.25]
  NGTKS = [0]


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
    # print(row)
    if row['mask'].split(':')[1] not in MASKS:
      continue
    if int(row['mask_size'].split(':')[1]) not in MSIZES:
      continue
    if int(row['ngtk'].split(':')[1]) not in NGTKS:
      continue
    if float(row['init_lr'].split(':')[1]) not in LRS:
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
  print(list(cdf))

  def resort(vlist):
    tmp = sorted([(float(v.split(':')[-1]), v) for v in vlist])
    return [v for (_, v) in tmp]

  INIT_LR = cdf['init_lr'].unique()
  print(f"initial learning rates: {INIT_LR}")
  INIT_LR = resort(INIT_LR)
  print(f"initial learning rates (resorted): {INIT_LR}")
  LRDRATE = cdf['lr_decay_rate'].unique()
  print(f"LR decay rates: {LRDRATE}")
  LRDRATE = resort(LRDRATE)
  print(f"LR decay rates (resorted): {LRDRATE}")
  NHEADS = cdf['nheads'].unique()
  print(f"# heads per blocks: {NHEADS}")
  NHEADS = resort(NHEADS)
  print(f"# heads per blocks (resorted): {NHEADS}")
  NBLOCKS = cdf['nblocks'].unique()
  print(f"# blocks: {NBLOCKS}")
  NBLOCKS = resort(NBLOCKS)
  print(f"# blocks (resorted): {NBLOCKS}")

  mopt = ''
  nvals = 0
  for olist, oname in zip(
      [INIT_LR, LRDRATE, NHEADS, NBLOCKS],
      ['init_lr', 'lr_decay_rate', 'nheads', 'nblocks']
  ):
    if len(olist) > 1:
      assert mopt == ''
      mopt = oname
      nvals = len(olist)
  if mopt == '':
    mopt = 'init_lr'
    nvals = 1

  methods = []
  mstr = lambda m, s, lr, dr, nb, nh: (
    f"{m.split(':')[-1]}({s.split(':')[-1]}):{lr.split(':')[-1]}:{dr.split(':')[-1]}:{nb.split(':')[-1]}:{nh.split(':')[-1]}"
    .replace('-1','')
    .replace('()','')
    .replace('none', 'full')
  )

  for (m, s, lr, dr, nb, nh) in product(MASKS, MSIZES, INIT_LR, LRDRATE, NBLOCKS, NHEADS):
    if (m == 'none' and s != -1) or (s == -1 and m != 'none'):
      continue
    methods += [mstr(m, str(s), str(lr), str(dr), str(nb), str(nh))]
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
  print('color dictionary:\n', cdict)
  nplotcols = nvals # len(MASKS)
  nplotrows = 1
  statssets = ['tr', 'va']
  fig, axs = plt.subplots(
    nplotrows, nplotcols, sharex=True, sharey=args.sharey,
    figsize=(2.5*nplotcols + 1, 2.5 * nplotrows)
  )

  groups = cdf.groupby(['mask', 'mask_size', 'init_lr', 'lr_decay_rate', 'nblocks', 'nheads'])
  print(f"Found {groups.ngroups} groups of results")

  def plot_set(
      ax, df, label, color,
      pltcol=f'tr-{args.metric}',
      repcol=f'va-{args.metric}',
      higher=args.metric=='ac'):
    print(f"Plotting {df.shape} lines in {color} for {label} ...")
    aggstats = []
    best_rep_val, best_rep_iter, last_rep_val, cvg_iter = [], [], [], []
    for idx, (i, r) in enumerate(df.iterrows()):
      print(f"--> rep {idx+1}/{df.shape[0]}")
      lab = mstr(r['mask'], r['mask_size'], r['init_lr'], r['lr_decay_rate'], r['nblocks'], r['nheads'])
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
    try:
      allstats = np.array(aggstats)
    except Exception as e:
      max_size = np.max([aaa.shape[0] for aaa in aggstats])
      def_idx = np.min([i for i, aaa in enumerate(aggstats) if aaa.shape[0] == max_size])
      rlist = []
      for i in range(len(aggstats)):
        if aggstats[i].shape[0] == max_size:
          continue
        vals = aggstats[i].values
        print(f"extending {i+1}-th rep from {len(vals)} -> {max_size}")
        print(f" ... using stats from {def_idx}")
        aggstats[i] = pd.Series(np.append(vals, aggstats[def_idx].values[len(vals):max_size]))
        print(aggstats[i].shape)
        rlist += [i]
      print(f' --- post fix --- ')
      for aaa in aggstats:
        print(aaa.shape, type(aaa))
      allstats = np.array(aggstats)
    median = np.percentile(allstats, q=50, axis=0)
    lqr = np.percentile(allstats, q=25, axis=0)
    uqr = np.percentile(allstats, q=75, axis=0)
    ax.plot(
      np.arange(median.shape[0]),
      median,
      color=color,
      linewidth=2.5,
      label=label.upper(),
    )
    ax.fill_between(
      np.arange(median.shape[0]),
      lqr, uqr,
      color=color,
      alpha=0.2,
    )
    return retdict, median

  def lab2opt(lab, opt):
    vals = lab.split(':')
    assert len(vals) == 5
    opt2idx = {
      'mask': 0,
      'init_lr': 1,
      'lr_decay_rate': 2,
      'nblocks': 3,
      'nheads': 4,
    }
    assert opt in opt2idx
    return vals[opt2idx[opt]]

  resdict = {}
  meddict = {}
  for m in MASKS:
    for s in MSIZES:
      if s == -1 and m != 'none':
        continue
      if s != -1 and m == 'none':
        continue
      print(m, s)

      pidx = 0
      for (lr, dr, nb, nh) in product(INIT_LR, LRDRATE, NBLOCKS, NHEADS):
        label = mstr(m, str(s), str(lr), str(dr), str(nb), str(nh))
        print(label)
        setdf = [gdf for ((mm, ss, ll, dd, bb, hh), gdf) in groups if mstr(mm, ss, ll, dd, bb, hh) == label]
        assert len(setdf) <= 1, (
          f"[mask:{m} msize:{s} initlr:{lr} drate:{dr} label:{label} B:{nb} H:{nh}]"
          f" got: {len(setdf)} sets:\n {setdf}"
        )
        if len(setdf) == 0:
          print(
            f"[mask:{m} msize:{s} initlr:{lr} drate:{dr} label:{label} B:{nb} H:{nh}]"
            f" got: {len(setdf)} sets:\n {setdf}"
          )
          continue
        print(f"---> plotting set {label}")
        if m == 'none':
          ax = axs[pidx] if nplotcols > 1 else axs
          ax.minorticks_on()
          ax.grid(which='major', axis='both', alpha=0.9)
          ax.grid(which='minor', axis='both', alpha=0.2)
          ax.set_title(f"{mopt.replace('_', ' ').upper()}:{lab2opt(label, mopt)}")
          ax.set_xlabel("Epochs")
        resdict[label], meddict[label] = plot_set(
          axs[pidx] if nplotcols > 1 else axs,
          setdf[0],
          label.split(':')[0],
          cdict[label.split(':')[0]]
        )
        pidx += 1

  # for ax in axs:
  ax = axs[0] if nplotcols > 1 else axs
  ax.set_ylabel("Training Loss")
  ax.legend(ncol=1, fontsize=10, loc='best', handlelength=1)

  fig.tight_layout()
  lfile = os.path.join(args.output_dir, OUTFILE)
  print(f"Saving figures in {lfile} ...")
  fig.savefig(lfile)

  print(meddict.keys())
  for k1 in meddict.keys():
    if "full:" not in k1:
      continue
    k2 = k1.replace("full:", "topk(5):")
    v1 = meddict[k1]
    v2 = meddict[k2]
    minlen = np.min([v1.shape[0], v2.shape[0]])
    print(k1, v1.shape, k2, v2.shape, f"upto epoch {minlen}")
    maxabs, maxrel, maiter, mriter = 0, 0, -1, -1
    maxspeedup, msiter = 0, -1
    for i in range(minlen):
      a = v1[i] - v2[i]
      r = a * 100. / v1[i]
      if i == 0:
        print(f"After epoch 1: {a:0.3f}, {r:0.3f}")
      if a > maxabs:
        maxabs = a
        maiter = i
      if r > maxrel:
        maxrel = r
        mriter = i
      speedup = np.sum(v2[:i] <= v1[i]) * 100. / (i+1)
      if (speedup > maxspeedup):
        maxspeedup = speedup
        msiter = i
    print(f"Max abs {maxabs:0.3f} at epoch {maiter + 1}")
    print(f"Max rel {maxrel:0.3f} at epoch {mriter + 1}")
    print(f"Max speedup {maxspeedup:0.3f} at epoch {msiter + 1}")
