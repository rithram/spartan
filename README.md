# Transformers Learn Faster with Semantic Focus

> [!NOTE]
>  ⚠️ 🚧 **Under construction** <br>
Code, results and plotting scripts coming soon!

Experiments with transformers with various sparse attention mechanisms.

## Environment setup

### Installation

Assuming CUDA is properly setup on the machine, we will be using python version 3.11

```
> cd spartan
> conda create -n spartan
> conda activate spartan
> conda install python=3.11 pip>25.0
> conda install cudatoolkit -c anaconda  # <== OPTIONAL: if we have access to a GPU
> pip install -r requirements.txt
```

## Experimental details

- Data: [data.md](./data.md)
- Training runs: [train.md](./train.md)
- Figures: [pltcmds.md](./pltcmds.md)
