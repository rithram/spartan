#!/bin/bash

NTRAIN=5000
NVAL=2000
LEN=40
LSTEP=0
NEMB=64
NHEAD=1
NBLOCK=5
LR=0.1
BSZ=25
MLPA=$1

DATA="missdup"
NEPOCHS=250
ODIR="results/missdup"
CDIR="checkpoints/missdup"
for NUM in {1..10}; do
  SEED=$(sed "${NUM}q;d" random_seeds)
  echo "RUNNING REP ${NUM}/10 WITH SEED ${SEED} ..."
  python ch.py --ntrain ${NTRAIN} --nval ${NVAL} \
       --min_len ${LEN} --max_len ${LEN} --len_step ${LSTEP} --data ${DATA} \
       -e ${NEMB} -m ${NEMB} -B ${NBLOCK} -H ${NHEAD} \
       -M none -E ${NEPOCHS} --seed ${SEED} \
       --bsz ${BSZ} -l ${LR} \
       -O ${ODIR} -C ${CDIR} --mlpa ${MLPA}
  for MS in 5 9; do
    for MASK in windowed blklocal topk; do
      python ch.py --ntrain ${NTRAIN} --nval ${NVAL} \
           --min_len ${LEN} --max_len ${LEN} --len_step ${LSTEP} --data ${DATA} \
           -e ${NEMB} -m ${NEMB} -B ${NBLOCK} -H ${NHEAD} \
           -M ${MASK} --mask_size ${MS} \
           -E ${NEPOCHS} --seed ${SEED} \
           --bsz ${BSZ} -l ${LR} \
           -O ${ODIR} -C ${CDIR} --mlpa ${MLPA}
    done
    for NGTK in 1 3; do
      for MASK in windowed blklocal; do
        python ch.py --ntrain ${NTRAIN} --nval ${NVAL} \
             --min_len ${LEN} --max_len ${LEN} --len_step ${LSTEP} \
             --data ${DATA} \
             -e ${NEMB} -m ${NEMB} -B ${NBLOCK} -H ${NHEAD} \
             -M ${MASK} --mask_size ${MS} \
             -E ${NEPOCHS} --seed ${SEED} \
             --bsz ${BSZ} -l ${LR} \
             -O ${ODIR} -C ${CDIR} \
             --ngtk ${NGTK} --mlpa ${MLPA}
      done
    done
  done
done
