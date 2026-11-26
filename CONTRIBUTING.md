# Contributing to PAZ

Thank you for helping improve PAZ. This guide covers the development setup,
the pull-request flow and the code style we review against.

## Development install

PAZ requires Python >= 3.10 and runs Keras 3 on the JAX backend.

```bash
git clone https://github.com/oarriaga/paz.git
cd paz
git checkout paz-jax
pip install -e .
export KERAS_BACKEND=jax
```

Install the JAX build that matches your accelerator before installing PAZ;
the default wheel from PyPI is CPU-only.

## Running tests

Tests live beside the module they cover as `*_test.py`, plus a `tests/` tree.

```bash
pytest paz/graphics/renderer_test.py   # nearest target first
pytest paz/ tests/                     # full suite
```

Tests that download weights are opt-in: they are skipped unless their
environment variable is set (for example `GEMMA4_WEIGHTS_TEST=1`), so the
default suite runs offline on CPU.

## Pull requests

- Branch per task off `paz-jax` (the integration branch until 1.0 is
  released) and name it for the task, e.g. `fix/ssd-prior-boxes` or
  `feat/cityscapes-loader`.
- Keep one logical change per pull request so reviews and reverts stay clean.
- Add or update tests for the behavior you change, and run the nearest pytest
  target before opening the pull request.
- State the verification results in the pull-request body: which tests you
  ran and what they reported.
- Keep commit messages and pull-request text short: what changed and why.
- Address review feedback with follow-up commits on the same branch.
- The maintainer reviews and merges; please do not merge your own pull
  request.

## Code style

- JAX-first: use JAX arrays and `import jax.numpy as jp` in tensor code, and
  keep functions compatible with `jax.jit` and `jax.vmap`.
- Prefer pure functions and immutable data; use namedtuples for structured
  values instead of dictionaries.
- Write short functions, often around 6 lines, ordered from high-level to
  low-level.
- Use full short English names in snake_case; `num` for counts and common
  short names such as `x`, `H` and `W` are fine.
- Keep lines within 80 columns.
- Do not add type hints.
- Functions that need randomness take a JAX PRNG key as their first argument.
- Reuse existing `paz` helpers (`paz.image`, `paz.draw`, `paz.SO3`,
  `paz.graphics`, ...) before adding new ones.
- Build Keras models with the functional API.
- Keep comments rare; explain why, not what.
- Tests assert behavior against an independent expectation (a known value, a
  reference implementation or a snapshot), not against the code under test.
