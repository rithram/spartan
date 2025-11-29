#!/bin/bash

TRAIN="data/listops/D10-A10-l500-L600-5k-2k-2k/basic_train.tsv"
VAL="data/listops/D10-A10-l500-L600-5k-2k-2k/basic_val.tsv"
NEMB=64
NHEAD=1
NBLOCK=10
LR=1.0
DLR=0.99
DR=0.01
BSZ=25
NEPOCHS=200
ODIR="results/listops"
CDIR="checkpoints/listops"
MLPA=$1

for NUM in {1..10}; do
  SEED=$(sed "${NUM}q;d" random_seeds)
  echo "RUNNING REP ${NUM}/10 WITH SEED ${SEED} ..."
  python listops.py \
         -T ${TRAIN} -V ${VAL} --nocls \
         -e ${NEMB} -m ${NEMB} -B ${NBLOCK} -H ${NHEAD} \
         -d ${DR} -l ${LR} -D ${DLR} -b ${BSZ} -E ${NEPOCHS} \
         --seed ${SEED} \
         -O ${ODIR} -C ${CDIR} --mlpa ${MLPA}
  for MS in 5 9; do
    for MASK in windowed blklocal topk; do
      python listops.py \
             -T ${TRAIN} -V ${VAL} --nocls \
             -e ${NEMB} -m ${NEMB} -B ${NBLOCK} -H ${NHEAD} -M ${MASK} --mask_size ${MS} \
             -d ${DR} -l ${LR} -D ${DLR} -b ${BSZ} -E ${NEPOCHS} \
             --seed ${SEED} \
             -O ${ODIR} -C ${CDIR}  --mlpa ${MLPA}
    done
    for NGTK in 1 3; do
      for MASK in windowed blklocal; do
        python listops.py \
               -T ${TRAIN} -V ${VAL} --nocls \
               -e ${NEMB} -m ${NEMB} -B ${NBLOCK} -H ${NHEAD} -M ${MASK} --mask_size ${MS} \
               -d ${DR} -l ${LR} -D ${DLR} -b ${BSZ} -E ${NEPOCHS} \
               --seed ${SEED} \
               --ngtk ${NGTK} \
               -O ${ODIR} -C ${CDIR} --mlpa ${MLPA}
      done
    done
  done
done
