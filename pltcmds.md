# Plots

We are assuming that all the results are saved as in the provided scripts to generate results for all the tasks. We will be saving all the figures in the following `paperfigs` directory.

```
> mkdir paperfigs
```

## Learning & Generalization:

### Compact Results in Figure 1

#### Figure 1(a): List Operations + ReLU

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig1atop --metric ce --compact --cmask_size 5 --split tr
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig1abot --metric ac --compact --cmask_size 5 --split va
```

#### Figure 1(b): Even Pairs + ReLU

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig1btop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig1bbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```

#### Figure 1(c): Missing Duplicates + ReLU

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig1ctop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig1cbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```

### Compact Results in Figures 7-8

Note that Figures 1(a), (b), (c) are the same as Figures 7(a), (c), (d) respectively.

#### Figure 7(b): Parity + ReLU


Top row --- training cross-entropy loss:
```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig7btop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig7bbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```


#### Figure 8(a): Stack Manipulation + ReLU


Top row --- training cross-entropy loss:
```
python trfigs.py -I results/stackman \
  -R stackman-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:200__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8atop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/stackman \
  -R stackman-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:200__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8abot --metric ac --compact --cmask_size 5 --split va  --nolegend
```


#### Figure 8(b): Modular Arithmetic with Brackets + ReLU


Top row --- training cross-entropy loss:
```
python trfigs.py -I results/mab \
  -R mab-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8btop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/mab \
  -R mab-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8bbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```


#### Figure 8(c): Solve Equation  + ReLU


Top row --- training cross-entropy loss:
```
python trfigs.py -I results/soleq \
  -R soleq-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8ctop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/soleq \
  -R soleq-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8cbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```

#### Figure 8(d): Cycle Navigation + ReLU


Top row --- training cross-entropy loss:
```
python trfigs.py -I results/cycnav \
  -R cycnav-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:750__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8dtop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/cycnav \
  -R cycnav-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:750__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig8dbot --metric ac --compact --cmask_size 5 --split va --nolegend
```

### Extended Results with Varying Mask Sizes

#### Figure 9(a): Training loss with List Operations + ReLU

```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig9a --metric ce --split tr
```

#### Figure 9(b): Training loss with Parity + ReLU

```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig9b --metric ce --split tr
```

#### Figure 9(c): Training loss with Even Pairs + ReLU

```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig9c --metric ce --split tr
```

#### Figure 9(d): Training loss with Missing Duplicates + ReLU

```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig9d --metric ce --split tr
```


#### Figure 10(a): Training accuracy with List Operations + ReLU

```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig10a --metric ac --split tr
```

#### Figure 10(b): Training accuracy with Parity + ReLU

```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig10b --metric ac --split tr
```

#### Figure 10(c): Training accuracy with Even Pairs + ReLU

```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig10c --metric ac --split tr
```

#### Figure 10(d): Training accuracy with Missing Duplicates + ReLU

```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig10d --metric ac --split tr
```

#### Figure 11(a): Training loss with Stack Manipulation + ReLU

```
python trfigs.py -I results/stackman \
  -R stackman-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:200__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig11a --metric ce --split tr
```

#### Figure 11(b): Training loss with Modular Arithmetic + ReLU

```
python trfigs.py -I results/mab \
  -R mab-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig11b --metric ce --split tr
```

#### Figure 11(c): Training loss with Solve Equation + ReLU

```
python trfigs.py -I results/soleq \
  -R soleq-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig11c --metric ce --split tr
```

#### Figure 11(d): Training loss with Cycle Navigation + ReLU

```
python trfigs.py -I results/cycnav \
  -R cycnav-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:750__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig11d --metric ce --split tr
```

#### Figure 12(a): Training accuracy with Stack Manipulation + ReLU

```
python trfigs.py -I results/stackman \
  -R stackman-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:200__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig12a --metric ac --split tr
```

#### Figure 12(b): Training accuracy with Modular Arithmetic + ReLU

```
python trfigs.py -I results/mab \
  -R mab-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig12b --metric ac --split tr
```

#### Figure 12(c): Training accuracy with Solve Equation + ReLU

```
python trfigs.py -I results/soleq \
  -R soleq-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig12c --metric ac --split tr
```

#### Figure 12(d): Training accuracy with Cycle Navigation + ReLU

```
python trfigs.py -I results/cycnav \
  -R cycnav-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:750__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig12d --metric ac --split tr
```

## Effect of Activation Functions in the MLP Block

### Compact Results in Figure 2

#### Figure 2(a): List Operations + GELU

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig2atop --metric ce --compact --cmask_size 5 --split tr
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig2abot --metric ac --compact --cmask_size 5 --split va
```

#### Figure 2(b): Even Pairs + GELU

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig2btop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig2bbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```

#### Figure 2(c): Missing Duplicates + GELU

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig2ctop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig2cbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```

#### Figure 2(d): List Operations + Mish

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig2dtop --metric ce --compact --cmask_size 5 --split tr
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig2dbot --metric ac --compact --cmask_size 5 --split va
```

#### Figure 2(e): Even Pairs + Mish

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig2etop --metric ce --compact --cmask_size 5 --split tr
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig2ebot --metric ac --compact --cmask_size 5 --split va
```

#### Figure 2(f): Missing Duplicates + Mish

Top row --- training cross-entropy loss:
```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig2ftop --metric ce --compact --cmask_size 5 --split tr  --nolegend
```

Bottom row --- heldout accuracy:
```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig2fbot --metric ac --compact --cmask_size 5 --split va  --nolegend
```


### Extended Results with Varying Mask Sizes

#### Figure 13(a): Training loss with List Operations + GELU

```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig13a --metric ce --split tr
```

#### Figure 13(b): Training loss with Parity + GELU

```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig13b --metric ce --split tr
```

#### Figure 13(c): Training loss with Even Pairs + GELU

```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig13c --metric ce --split tr
```

#### Figure 13(d): Training loss with Missing Duplicates + GELU

```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig13d --metric ce --split tr
```

#### Figure 14(a): Training accuracy with List Operations + GELU

```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig14a --metric ac --split tr
```

#### Figure 14(b): Training accuracy with Parity + GELU

```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig14b --metric ac --split tr
```

#### Figure 14(c): Training accuracy with Even Pairs + GELU

```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig14c --metric ac --split tr
```

#### Figure 14(d): Training accuracy with Missing Duplicates + GELU

```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig14d --metric ac --split tr
```

#### Figure 15(a): Training loss with List Operations + Mish

```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig15a --metric ce --split tr
```

#### Figure 15(b): Training loss with Parity + Mish

```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig15b --metric ce --split tr
```

#### Figure 15(c): Training loss with Even Pairs + Mish

```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig15c --metric ce --split tr
```

#### Figure 15(d): Training loss with Missing Duplicates + Mish

```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig15d --metric ce --split tr
```

#### Figure 16(a): Training accuracy with List Operations + Mish

```
python trfigs.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig16a --metric ac --split tr
```

#### Figure 16(b): Training accuracy with Parity + Mish

```
python trfigs.py -I results/parity \
  -R parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig16b --metric ac --split tr
```

#### Figure 16(c): Training accuracy with Even Pairs + Mish

```
python trfigs.py -I results/ueqpairs \
  -R ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig16c --metric ac --split tr
```

#### Figure 16(d): Training accuracy with Missing Duplicates + Mish

```
python trfigs.py -I results/missdup \
  -R missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:*__mlpa:mish.csv \
  -O paperfigs/ -N fig16d --metric ac --split tr
```

#### Figure 17(a): Training loss with Stack Manipulation + GELU

```
python trfigs.py -I results/stackman \
  -R stackman-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:200__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig17a --metric ce --split tr
```

#### Figure 17(b): Training loss with Modular Arithmetic + GELU

```
python trfigs.py -I results/mab \
  -R mab-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig17b --metric ce --split tr
```

#### Figure 17(c): Training loss with Solve Equation + GELU

```
python trfigs.py -I results/soleq \
  -R soleq-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig17c --metric ce --split tr
```

#### Figure 17(d): Training loss with Cycle Navigation + GELU

```
python trfigs.py -I results/cycnav \
  -R cycnav-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:750__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig17d --metric ce --split tr
```

#### Figure 18(a): Training loss with Stack Manipulation + GELU

```
python trfigs.py -I results/stackman \
  -R stackman-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:200__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig18a --metric ac --split tr
```

#### Figure 18(b): Training loss with Modular Arithmetic + GELU

```
python trfigs.py -I results/mab \
  -R mab-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig18b --metric ac --split tr
```

#### Figure 18(c): Training loss with Solve Equation + GELU

```
python trfigs.py -I results/soleq \
  -R soleq-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:600__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig18c --metric ac --split tr
```

#### Figure 18(d): Training loss with Cycle Navigation + GELU

```
python trfigs.py -I results/cycnav \
  -R cycnav-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:750__seed:*__mlpa:gelu.csv \
  -O paperfigs/ -N fig18d --metric ac --split tr
```

## Varying Architectural Hyperparameters

### Figure 3(a): Varying number of transformer layers (or blocks)

```
python trfigs-lr.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:*__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv
  -O paperfigs/ -N fig3a --metric ce --sharey
```

### Figure 3(b): Varying number of attention heads in each transformer layer (or block)

```
python trfigs-lr.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:*__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig3b --metric ce --sharey
```

## Varying Optimization Hyperparameters


### Figure 3(c): Varying initial learning rate

```
python trfigs-lr.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:*__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig3c --metric ce --sharey
```

### Figure 3(d): Varying learning rate decay rate

```
python trfigs-lr.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:*__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu.csv \
  -O paperfigs/ -N fig3d --metric ce --sharey
```

### Figure 3(e): Varying optimizer

```
python trfigs-lr.py -I results/listops \
  -R D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:*__lr_decay_rate:0.99999__bsz:25__nepochs:200__nocls:True__seed:*__mlpa:relu__adam:True.csv \
  -O paperfigs/ -N fig3e --metric ce
```

## Preliminary Experiments for Next-Token-Prediction with Penn Tree Bank

### Figure 19

```
python trfigs-lr.py -I  results/ptb \
  -R "ptb__demb:32__dmlp:128__nblocks:*__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:128__nepochs:50__seed:*__mlpa:relu__sp_vocab_size:4096__sp_model_type:unigram.csv"
  -O paperfigs/ -N fig19 --metric ce
```

### Figure 20

```
python trfigs-lr.py -I results/ptb \
  -R "ptb__demb:128__dmlp:512__nblocks:*__nheads:4__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:128__nepochs:120__seed:*__mlpa:relu__sp_vocab_size:10000__sp_model_type:bpe.csv" \
  -N fig20 --metric ce -O paperfigs/
```

## Loss Surfaces and Estimated Lipschitz Constants

### Generating loss surface

We will be utilizing the following checkpoints to generate the loss landscapes. These checkpoint would have been created if the per-task scripts were executed successfully:

```
List Operations + ReLU:
- Full attention: checkpoints/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu_last.pt
- Top-k attention with k=5: checkpoints/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu_last.pt

List Operations + GELU:
- Full attention: checkpoints/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:gelu_last.pt
- Top-k attention with k=5: checkpoints/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:gelu_last.pt

Even Pairs + ReLU:
- Full attention: checkpoints/ueqpairs/ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:140439608__mlpa:relu_last.pt
- Top-k attention with k=5: checkpoints/ueqpairs/ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:140439608__mlpa:relu_last.pt

Even Pairs + GELU:
- Full attention: checkpoints/ueqpairs/ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:140439608__mlpa:gelu_last.pt
- Top-k attention with k=5: checkpoints/ueqpairs/ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:140439608__mlpa:gelu_last.pt

Missing Duplicates + ReLU:
- Full attention: checkpoints/missdup/missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:140439608__mlpa:relu_last.pt
- Top-k attention with k=5: checkpoints/missdup/missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:140439608__mlpa:relu_last.pt

Missing Duplicates + GELU:
- Full attention: checkpoints/missdup/missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:140439608__mlpa:gelu_last.pt
- Top-k attention with k=5: checkpoints/missdup/missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:140439608__mlpa:gelu_last.pt

Parity + ReLU:
- Full attention: checkpoints/parity/parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:140439608__mlpa:relu_last.pt
- Top-k attention with k=5: checkpoints/parity/parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:140439608__mlpa:relu_last.pt

Parity + GELU:
- Full attention: checkpoints/parity/parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:140439608__mlpa:gelu_last.pt
- Top-k attention with k=5: checkpoints/parity/parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:140439608__mlpa:gelu_last.pt
```

To generate the loss landscapes for each of these checkpoints, we will make use of the script `gen-loss-landscape.sh` and provide each of the above checkpoints as an argument one at a time. Note that the loss landscape generation can take some time. We will be saving the results in the `lsurfs` directory:

```
mkdir lsurfs
for task in listops ueqpairs missdup parity; do mkdir -p lsurfs/${task}; done
```

As an example, we can generate the loss landscape for the List Operations task and the full-attention model and ReLU activation as follows:
```
bash gen-loss-landscape.sh checkpoints/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu_last.pt
```
The result for the above run would be found in:
```
lsurfs/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv
```

### Plotting loss landscapes and heatmaps (Figures 5, 23-26)

#### ReLU activation (Figures 5, 23, 24)

##### List Operations

The following command plots the loss landscapes in the form of contour plots (figure 5(a) top & middle rows, figure 23(a)) and heatmaps (figure 24(a)):
```
python ll-plot.py \
  --lldata "lsurfs/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

##### Parity

The following command plots the loss landscapes in the form of contour plots (figure 23(b)) and heatmaps (figure 24(b)):
```
python ll-plot.py \
  --lldata "lsurfs/parity/parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

##### Even Pairs

The following command plots the loss landscapes in the form of contour plots (figure 5(b) top & middle rows, figure 23(c)) and heatmaps (figure 24(c)):
```
python ll-plot.py \
  --lldata "lsurfs/ueqpairs/ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

##### Missing Duplicates

The following command plots the loss landscapes in the form of contour plots (figure 5(c) top & middle rows, figure 23(d)) and heatmaps (figure 24(d)):
```
python ll-plot.py \
  --lldata "lsurfs/missdup/missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

#### GELU activation (Figures 25, 26)

##### List Operations

The following command plots the loss landscapes in the form of contour plots (figure 25(a)) and heatmaps (figure 26(a)):
```
python ll-plot.py \
  --lldata "lsurfs/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:gelu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

##### Parity

The following command plots the loss landscapes in the form of contour plots (figure 25(b)) and heatmaps (figure 26(b)):
```
python ll-plot.py \
  --lldata "lsurfs/parity/parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:140439608__mlpa:gelu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

##### Even Pairs

The following command plots the loss landscapes in the form of contour plots (figure 25(c)) and heatmaps (figure 26(c)):
```
python ll-plot.py \
  --lldata "lsurfs/ueqpairs/ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:140439608__mlpa:gelu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

##### Missing Duplicates

The following command plots the loss landscapes in the form of contour plots (figure 25(d)) and heatmaps (figure 26(d)):
```
python ll-plot.py \
  --lldata "lsurfs/missdup/missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:140439608__mlpa:gelu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/ --nlevels 100 --masks none:topk
```

### Estimated Lipschitz constants (Figures 5, 27)

The following command

#### List Operations

The following command plots the estimated Lipschitz constants using the above loss landscapes (figure 5(a) bottom row, figure 27(a)):
```
python lgrads.py \
  --lldata "lsurfs/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/  --masks none:topk --uperc 99 --xmax 0.7 --tstep 0.01
```

#### Parity

The following command plots the estimated Lipschitz constants using the above loss landscapes (figure 27(b)):
```
python lgrads.py \
  --lldata "lsurfs/parity/parity-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:1000__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/  --masks none:topk --uperc 99 --xmax 0.7 --tstep 0.01
```

#### Even Pairs

The following command plots the estimated Lipschitz constants using the above loss landscapes (figure 5(b) bottom row, figure 27(c)):
```
python lgrads.py \
  --lldata "lsurfs/ueqpairs/ueqpairs-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:100__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/  --masks none:topk --uperc 99 --xmax 0.7 --tstep 0.01
```

#### Missing Duplicates

The following command plots the estimated Lipschitz constants using the above loss landscapes (figure 5(c) bottom row, figure 27(d)):
```
python lgrads.py \
  --lldata "lsurfs/missdup/missdup-l40-s0-L40-5000-2000-0__demb:64__dmlp:64__nblocks:5__nheads:1__mask:*__mask_size:*__dropout:0.01__init_lr:0.1__lr_decay_rate:0.9995__bsz:25__nepochs:250__seed:140439608__mlpa:relu__aub:1.0__nsteps:200__pseed:11111__lsurf.csv" \
  --output paperfigs/  --masks none:topk --uperc 99 --xmax 0.7 --tstep 0.01
```

## Bound comparisons:

### Figures 21

```
python bounds.py -O paperfigs/ --precision 0.0001
```

### Figures 22

```
python bounds.py -O paperfigs/ --precision 0.0001 --xcomp
```

### Table 6

To generate the results in Table 6, we find need to compute the relevant statistics using an existing checkpoint (the same we have used to generate the loss landscapes). The following commands generate the relevant statistics for each trained model. First, we will create the directories to store the statistics:

```
mkdir bounds
for task in listops ueqpairs missdup parity; do mkdir -p bounds/${task}; done
```

As an example, we can compute the full dispersion statistics of the model trained for the List Operations task with the following command:
```
python was.py \
  --checkpoint checkpoints/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:none__mask_size:-1__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu_last.pt
  --output bounds/ --batch_size 100 --pseed 1111 --aub 0.0 --nsteps 1 --nbatches 10
```

The heavy-hitter dispersion and separation of the model trained for the List Operations task can be computed with the following command:
```
python was.py \
  --checkpoint checkpoints/listops/D10-A10-l500-L600-5k-2k-2k__demb:64__dmlp:64__nblocks:10__nheads:1__mask:topk__mask_size:5__dropout:0.01__init_lr:1.0__lr_decay_rate:0.99__bsz:25__nepochs:200__nocls:True__seed:140439608__mlpa:relu_last.pt
  --output bounds/ --batch_size 100 --pseed 1111 --aub 0.0 --nsteps 1 --nbatches 10
```

These statistics can be similarly generated for the remaining tasks using their corresponding checkpoints. Once the statistics are computed in the `bounds/.../` directories, we can print the numbers in Table 6 by running the following:
```
python bounds-lhs.py --indir bounds/
```