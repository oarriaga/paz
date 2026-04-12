The eigenfaces example in this branch is pure JAX and pure functions.

Scripts:

- `python examples/eigenfaces/eigenfaces.py`
- `python examples/eigenfaces/database.py`
- `python examples/eigenfaces/demo.py`

Saved files:

- `experiments/eigenfaces_state.npz`
- `database/database_state.npz`

Database images must follow:

```text
database/
  images/
    person_a/
      image_01.png
    person_b/
      image_01.png
```

The training script uses `FERPlus` by default through `paz.datasets.load`.
