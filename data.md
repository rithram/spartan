# Data for the Empirical Evaluations

## ListOps

The data in the `./data/listops/` directory is generated using code from the [Long Range Arena](https://github.com/google-research/long-range-arena) using the following procedure:

```
> git clone git@github.com:google-research/long-range-arena.git
> cd long-range-arena
> mkdir env
> virtualenv -p /usr/bin/python3.11 env
> source env/bin/activate
> pip install -r requirements.txt
> python lra_benchmarks/data/listops.py --max_args <NARGS> --max_depth <DEPTH> \
                                        --max_length <MAXLEN> --min_length <MINLEN> \
                                        --num_train_samples <NTRAIN> \
                                        --num_valid_samples <NVAL> \
                                        --num_test_samples <NTEST> \
                                        --output_dir D<DEPTH>-A<NARGS>-l<MINLEN>-L<MAXLEN>-<NTRAIN>-<NVAL>-<NTEST>
```

For our experiments, we used:
```
<NARGS> = 10
<DEPTH> = 10
<MINLEN> = 500
<MAXLEN> = 600
<NTRAIN> = 5000
<NVAL> = 2000
<NTEST> = 2000
```

## Neural Networks and Chomsky Hierarchy

The data for the tasks from the NNCH benchmark is generated based on the code from [Neural Networks Chomsky Hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/) benchmark based on the following scripts written in JAX that we ported to pytorch in `./datasets.py`:
- [parity task](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/tasks/regular/parity_check.py)
- [even pairs task](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/tasks/regular/even_pairs.py)
- [missing duplicates task](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/tasks/cs/missing_duplicate_string.py)
- [stack manipulation task](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/tasks/dcf/stack_manipulation.py)
- [cycle navigation task](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/tasks/regular/cycle_navigation.py)
- [modular arithmetic with brackets task](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/tasks/dcf/modular_arithmetic_brackets.py)
- [solve equation task](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/tasks/dcf/solve_equation.py)


## Penn Tree Bank Dataset

We utilize the PennTreeBank dataset available in `torchtext.datasets`. For the sentence-piece tokenizer, we also need the text data in file for training the model. For this reason, we also make the PTB train/test/validation files available in [`data/ptb/`](./data/ptb/).