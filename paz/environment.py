import os


def set_memory_fraction(value):
    value = str(value)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = value
    return value


def set_gpu_device(device):
    device = str(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    return device


def set_debug_nans(enabled=True):
    import jax

    jax.config.update("jax_debug_nans", enabled)
    return enabled


def set_platform(name):
    import jax

    jax.config.update("jax_platform_name", name)
    return name
