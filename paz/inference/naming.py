import re


_NAME_COUNTERS = {}


def _to_snake_case(value):
    if value is None:
        return ""
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def _callable_name(value):
    name = getattr(value, "__name__", None)
    if name is None:
        return "callable"
    if name == "<lambda>":
        return "lambda"
    return name


def build_auto_name(role, base=None):
    role = _to_snake_case(role)
    base = _to_snake_case(base)
    key = (role, base)
    count = _NAME_COUNTERS.get(key, 0)
    _NAME_COUNTERS[key] = count + 1
    if base:
        return f"{role}_{base}_{count}"
    return f"{role}_{count}"


def build_prior_name(distribution):
    base = distribution.__class__.__name__
    return build_auto_name("prior", base)


def build_latent_name(distribution_fn):
    base = _callable_name(distribution_fn)
    return build_auto_name("latent", base)


def build_observation_name(distribution_fn):
    base = _callable_name(distribution_fn)
    return build_auto_name("observation", base)
