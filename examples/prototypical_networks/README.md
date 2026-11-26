# Prototypical networks

Few-shot classification on Omniglot with prototypical networks, plus a
sample-complexity experiment over the number of training classes.

| Script | What it does |
| --- | --- |
| `train.py` | Trains the embedding network and evaluates within-alphabet and between-alphabet accuracy. |
| `run_sample_complexity_experiment.sh` | Runs `train.py` 20 times for `--train_classes` from 100 to 900 in steps of 50. |
| `plot.py` | Reads the `results.json` files of those runs and plots test accuracy against training classes. |

## Run

```bash
cd examples/prototypical_networks
python3 train.py --train_ways 60 --train_shots 5 --train_queries 5 \
    --epochs 5 --steps_per_epoch 2000
python3 plot.py --results_dir experiments --num_seeds 20
```

`train.py` writes the weights, a `.keras` model and `results.json` into a
run folder under `--root` (default `experiments`). `plot.py` writes
`--output_file` (default `omniglot_test-accuracy_vs_training-classes.pdf`).

## Requirements

Omniglot images at `--train_path` (default `omniglot/images_background/`)
and `--tests_path` (default `omniglot/images_evaluation/`). The dataset is
not shipped in this folder.
