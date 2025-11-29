#!/bin/bash

VSIZE=4096
MTYPE="unigram"
NEMB=32
NMLP=128
NHEAD=1
LR=1.0
DLR=0.99
DR=0.01
BSZ=128
NEPOCHS=50
ODIR="results/ptb"
CDIR="checkpoints/ptb"
SPDATA="data/ptb/ptb.train.txt"
SPMDIR="sp"
MLPA="relu"

MASK=$1
MSIZE=$2

NAME="LISTOPS-${MASK}-${MSIZE}"
echo $NAME

for NBLOCK in 8 16 32 64; do
  for TRIAL in {1..10}; do
    SEED=$(sed "${TRIAL}q;d" random_seeds)
    echo "RUNNING REP ${TRIAL}/10 WITH SEED ${SEED} ..."
    python ptb.py \
           --demb ${NEMB} --dmlp ${NMLP} --nblocks ${NBLOCK} --nheads ${NHEAD} \
           --mask ${MASK} --mask_size ${MSIZE} \
           --mlpa ${MLPA} \
           --dropout ${DR} --init_lr ${LR} --lr_decay_rate ${DLR} \
           --bsz ${BSZ} --nepochs ${NEPOCHS} \
           --sp_data ${SPDATA} --sp_model_dir ${SPMDIR} \
           --sp_vocab_size ${VSIZE} --sp_model_type ${MTYPE} \
           --seed ${SEED} \
           -O ${ODIR} -C ${CDIR}
    echo "REP ${TRIAL}/10 completed"
  done
done
