# Model-agnostic meta-learning

MAML in JAX for sinusoid regression and few-shot Omniglot classification.

| Script | What it does |
| --- | --- |
| `maml.py` | The MAML inner and outer update functions used by both training scripts. |
| `sinusoidal.py` | Sinusoid task sampling and plotting helpers. |
| `train_regression.py` | Meta-trains an MLP on sinusoid tasks and plots the results. |
| `train_classification.py` | Meta-trains on Omniglot N-way K-shot tasks and plots test tasks. |

## Run

```bash
python3 examples/model_agnostic_meta_learning/train_regression.py \
    --train_steps 20000 --tasks_per_batch 4 --shots_per_task 20
```

Other options of `train_regression.py`: `--learning_rate`,
`--inner_learning-rate`, `--num_steps`, `--seed`, `--min_x`, `--max_x`,
`--min_y`, `--max_y`, `--hidden_dims`.

Options of `train_classification.py`: `--learning_rate`,
`--inner_learning_rate`, `--train_steps`, `--tasks_per_batch`,
`--num_ways`, `--num_shots`, `--num_queries`, `--num_inner_steps`, `--seed`.

Both scripts show matplotlib figures with `plt.show()`; no files are
written.

## Requirements

- A display for the matplotlib figures.
- `train_classification.py` loads Omniglot through `paz.datasets.omniglot`.
