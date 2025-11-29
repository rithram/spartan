#!/bin/bash

TRAIN="data/listops/D10-A10-l500-L600-5k-2k-2k/basic_train.tsv"
VAL="data/listops/D10-A10-l500-L600-5k-2k-2k/basic_val.tsv"
NEMB=64
DR=0.01
BSZ=25
ODIR="results/listops"
CDIR="checkpoints/listops"
NGTK=0
MLPA="relu"
NEPOCHS=200
NHEAD=1
NBLOCK=10

MASK=$1
MSIZE=$2
# try values 0.66 1.0 1.5 2.25 (DLR=0.99)
LR=$3
# try values 0.9 0.99 0.999 0.9999 (LR=1.0)
DLR=$4

NAME="LISTOPS-${MASK}-${MSIZE}-${MLPA}-${TRIAL}-${LR}-${DLR}-${NEPOCHS}"
echo $NAME

for TRIAL in {1..10}; do
  SEED=$(sed "${TRIAL}q;d" random_seeds)
  echo "RUNNING REP ${TRIAL}/10 WITH SEED ${SEED} ..."
  python listops.py \
         -T ${TRAIN} -V ${VAL} --nocls \
         -e ${NEMB} -m ${NEMB} -B ${NBLOCK} -H ${NHEAD} \
         -M ${MASK} --mask_size ${MSIZE} --ngtk ${NGTK} \
         --mlpa ${MLPA} \
         -d ${DR} -l ${LR} -D ${DLR} -b ${BSZ} -E ${NEPOCHS} \
         --seed ${SEED} \
         -O ${ODIR} -C ${CDIR}
  echo "REP ${TRIAL}/10 completed"
done
