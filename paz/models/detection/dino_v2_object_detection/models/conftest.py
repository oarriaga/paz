# Release memory between tests. The real-weights parity tests each load a
# full RF-DETR checkpoint and build a JAX model; without freeing them the
# whole models tree run in one process exhausts RAM/GPU and aborts natively.

import gc

import pytest


@pytest.fixture(autouse=True)
def release_memory_after_test():
    yield
    gc.collect()
    clear_jax_caches()
    clear_torch_caches()


def clear_jax_caches():
    try:
        import jax
    except ImportError:
        return
    jax.clear_caches()


def clear_torch_caches():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
