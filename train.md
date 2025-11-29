# Training Run Details

All results and checkpoints will be saved in the following directories:
```
> mkdir results
> mkdir checkpoints
```

## Experiment hyperparameters

- 10 repetitions
- Transformer variants:
  - Standard
  - Banded with band size 5, 9 and global tokens 0, 1, 3
  - Blocklocal with block size 5, 9 and global tokens 0, 1, 3
  - topk with k = 5, 9

## Listops experiments

### Main training runs

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 10
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 1.0
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 200

```
> mkdir results/listops
> mkdir checkpoints/listops
> bash lra-lops.sh relu
> bash lra-lops.sh gelu
> bash lra-lops.sh mish
```

### Hyperparameter ablations

#### Number of transformer layers

Full attention model
```
> MASK="none"; MSIZE=-1; NHEAD=1; for NBLOCKS in 6 15 22; do bash lra-lops-bh.sh ${MASK} ${MSIZE} ${NBLOCKS} ${NHEAD}; done
```

Top-`k` attention model with `k = 5`
```
> MASK="topk"; MSIZE=5; NHEAD=1; for NBLOCKS in 6 15 22; do bash lra-lops-bh.sh ${MASK} ${MSIZE} ${NBLOCKS} ${NHEAD}; done
```

#### Number of attention heads per transformer layer

Full attention models
```
> MASK="none"; MSIZE=-1; NBLOCKS=10; for NHEAD in 2 4 8; do bash lra-lops-bh.sh ${MASK} ${MSIZE} ${NBLOCKS} ${NHEAD}; done
```

Top-`k` attention model with `k = 5`
```
> MASK="topk"; MSIZE=5; NBLOCKS=10; for NHEAD in 2 4 8; do bash lra-lops-bh.sh ${MASK} ${MSIZE} ${NBLOCKS} ${NHEAD}; done
```

#### Initial learning rate

Full attention models
```
> MASK="none"; MSIZE=-1; DLR=0.99; for LR in 0.66 1.5 2.25; do bash lra-lops-lr.sh ${MASK} ${MSIZE} ${LR} ${DLR}; done
```

Top-`k` attention model with `k = 5`
```
> MASK="topk"; MSIZE=5; DLR=0.99; for LR in 0.66 1.5 2.25; do bash lra-lops-lr.sh ${MASK} ${MSIZE} ${LR} ${DLR}; done
```

#### Learning rate decay

Full attention models
```
> MASK="none"; MSIZE=-1; LR=1.0; for DLR in 0.9 0.999 0.9999; do bash lra-lops-lr.sh ${MASK} ${MSIZE} ${LR} ${DLR}; done
```

Top-`k` attention model with `k = 5`
```
> MASK="topk"; MSIZE=5; LR=1.0; for DLR in 0.9 0.999 0.9999; do bash lra-lops-lr.sh ${MASK} ${MSIZE} ${LR} ${DLR}; done
```

#### Adam optimizer

Full attention models
```
> MASK="none"; MSIZE=-1; for LR in 0.00001 0.0001 0.001 0.01; do bash lra-lops-adam.sh ${MASK} ${MSIZE} ${LR}; done
```

Top-`k` attention model with `k = 5`
```
> MASK="topk"; MSIZE=5; for LR in 0.00001 0.0001 0.001 0.01; do bash lra-lops-adam.sh ${MASK} ${MSIZE} ${LR}; done
```



## Even pairs experiments

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 5
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 0.1
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 100

```
> mkdir results/ueqpairs
> mkdir checkpoints/ueqpairs
> bash nnch-even-pairs.sh relu
> bash nnch-even-pairs.sh gelu
> bash nnch-even-pairs.sh mish
```


## Parity experiments

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 5
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 0.1
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 1000

```
> mkdir results/parity
> mkdir checkpoints/parity
> bash nnch-parity.sh relu
> bash nnch-parity.sh gelu
> bash nnch-parity.sh mish
```

## Missing duplicates experiments

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 5
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 0.1
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 250

```
> mkdir results/missdup
> mkdir checkpoints/missdup
> bash nnch-missdup.sh relu
> bash nnch-missdup.sh gelu
> bash nnch-missdup.sh mish
```

## Stack Manipulation experiments

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 5
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 0.1
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 200

```
> mkdir results/stackman
> mkdir checkpoints/stackman
> bash nnch-stackman.sh relu
> bash nnch-stackman.sh gelu
> bash nnch-stackman.sh mish
```

## Modular Arithmetic with Brackets experiments

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 5
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 0.1
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 600

```
> mkdir results/mab
> mkdir checkpoints/mab
> bash nnch-mab.sh relu
> bash nnch-mab.sh gelu
> bash nnch-mab.sh mish
```

## Solve Equation experiments

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 5
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 0.1
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 600

```
> mkdir results/soleq
> mkdir checkpoints/soleq
> bash nnch-soleq.sh relu
> bash nnch-soleq.sh gelu
> bash nnch-soleq.sh mish
```

## Cycle Navigation experiments

#### Hyperparameters:
- Transformer architecture
  - Number of blocks: 5
  - Numbers of heads: 1
  - Embedding dimension: 64
  - MLP hidden layer dimension: 64
  - Dropout: 0.01
- Optimization
  - Learning rate: 0.1
  - Learning rate decay rate: 0.99
  - Batch size: 25
  - Epochs: 750

```
> mkdir results/cycnav
> mkdir checkpoints/cycnav
> bash nnch-cycnav.sh relu
> bash nnch-cycnav.sh gelu
> bash nnch-cycnav.sh mish
```

## Penn Tree Bank Experiments

First we will create the directories where the results and checkpoints will be saved:
```
mkdir results/ptb
mkdir checkpoints/ptb
```

We will be using the sentence-piece tokenizer. For this reason, we will need to train a sentence-piece model for each vocabulary size and `model_type`. These models will be need to be saved for all training runs. We will save them in the following:
```
mkdir sp
```

For the first set of experiments with a vocabulary size of 4096 with `model_type=unigram`, we will run the following commands for standard full attention transformer and top-`k` attention with `k=5`:
```
bash ptb-unigram.sh none -1
bash ptb-unigram.sh topk 5
```

For the second set of experiments with a larger vocabulary size of 10000 and byte-pair-encoding (that is `model_type=bpe` for sentence piece model), and training a larger transformer model, we do the following:
```
bash ptb-bpe.sh none -1
bash ptb-bpe.sh topk 5
```
