#!/bin/bash

CHKPT=$1
BSZ=128
AUB=1.0
NSTEPS=200
SEED=11111

echo "Computing loss surface:"
echo "- Checkpoint: ${CHKPT}"
echo "- Batch size: ${BSZ}"
echo "- Grid points per dimension: ${NSTEPS}"
echo "- Random direction seed: ${SEED}"

echo "Running loss surface computation ...."
python ll-test.py \
       --checkpoint ${CHKPT} --output lsurfs \
       --batch_size ${BSZ} --aub ${AUB} --nsteps ${NSTEPS} \
       --pseed ${SEED}
echo "Loss surface computation complete"
