import inspect

_DISTRIBUTION_BUILDERS = {}


def build_distribution_fn(fn_id, **kwargs):
    builder = _DISTRIBUTION_BUILDERS.get(fn_id)
    if builder is None:
        raise ValueError(f"Unknown distribution fn_id '{fn_id}'.")
    return builder(**kwargs)


def serializable(fn_id=None):
    def decorator(builder):
        name = builder.__name__ if fn_id is None else fn_id

        def wrapper(*args, **kwargs):
            bound = inspect.signature(builder).bind(*args, **kwargs)
            bound.apply_defaults()
            apply = builder(*args, **kwargs)
            return _with_spec(name, apply, bound.arguments)

        _register(name)(wrapper)
        return wrapper

    return decorator


def _register(fn_id):
    def decorator(builder):
        _DISTRIBUTION_BUILDERS[fn_id] = builder
        return builder
    return decorator


def _with_spec(fn_id, apply, kwargs):
    apply._paz_spec = {"fn_id": fn_id, "kwargs": kwargs}
    return apply


__all__ = ["build_distribution_fn", "serializable"]
