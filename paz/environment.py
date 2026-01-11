"""Environment configuration for JAX.

CRITICAL: Set environment variables BEFORE importing paz.

Usage
-----

Set environment variables before importing paz:

    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import paz  # JAX initializes with correct settings

Common environment variables:
    XLA_PYTHON_CLIENT_MEM_FRACTION: GPU memory fraction (e.g., "0.9")
    CUDA_VISIBLE_DEVICES: GPU device indices (e.g., "0" or "0,1,2")

Functions
---------

The functions below configure JAX behavior at runtime (after import):

set_debug_nans(enabled): Enable/disable JAX NaN debugging
set_platform(name): Set JAX platform ("cpu", "gpu", "tpu")
"""
import jax


def set_debug_nans(enabled=True):
    jax.config.update("jax_debug_nans", enabled)
    return enabled


def set_platform(name):
    jax.config.update("jax_platform_name", name)
    return name
