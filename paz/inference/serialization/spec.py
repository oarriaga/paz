import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent_space import LatentSpace
from paz.inference.types import SampleType

tfd = tfp.distributions
tfb = tfp.bijectors


def _latent_space_spec(latent_space, arrays, prefix):
    names = list(latent_space.names)
    bijectors = {}
    for name in names:
        bijectors[name] = _serialize_bijector(
            latent_space.bijectors[name], arrays, f"{prefix}_{name}"
        )
    return {"names": names, "bijectors": bijectors}


def _build_latent_space(spec, arrays):
    names = spec["names"]
    Sample = SampleType(names)
    bijectors = {
        name: _deserialize_bijector(spec["bijectors"][name], arrays)
        for name in names
    }
    return LatentSpace(names, bijectors, Sample)


def _serialize_bijector(bijector, arrays, prefix):
    if bijector is None:
        return {"kind": "Identity"}
    if isinstance(bijector, tfb.Identity):
        return {"kind": "Identity"}
    if isinstance(bijector, tfb.Shift):
        name = f"{prefix}_shift"
        arrays[name] = bijector.shift
        return {"kind": "Shift", "kwargs": {"shift": _ref(name)}}
    if isinstance(bijector, tfb.Scale):
        name = f"{prefix}_scale"
        arrays[name] = bijector.scale
        return {"kind": "Scale", "kwargs": {"scale": _ref(name)}}
    if isinstance(bijector, tfb.Sigmoid):
        spec = {"kind": "Sigmoid"}
        kwargs = {}
        if getattr(bijector, "low", None) is not None:
            name = f"{prefix}_low"
            arrays[name] = bijector.low
            kwargs["low"] = _ref(name)
        if getattr(bijector, "high", None) is not None:
            name = f"{prefix}_high"
            arrays[name] = bijector.high
            kwargs["high"] = _ref(name)
        if kwargs:
            spec["kwargs"] = kwargs
        return spec
    if isinstance(bijector, tfb.Softplus):
        spec = {"kind": "Softplus"}
        kwargs = {}
        hinge_softness = getattr(bijector, "hinge_softness", None)
        if hinge_softness is not None:
            name = f"{prefix}_hinge_softness"
            arrays[name] = hinge_softness
            kwargs["hinge_softness"] = _ref(name)
        if kwargs:
            spec["kwargs"] = kwargs
        return spec
    if isinstance(bijector, tfb.Exp):
        return {"kind": "Exp"}
    if isinstance(bijector, tfb.SoftmaxCentered):
        return {"kind": "SoftmaxCentered"}
    if isinstance(bijector, tfb.Invert):
        inner = _serialize_bijector(
            bijector.bijector, arrays, f"{prefix}_invert"
        )
        return {"kind": "Invert", "bijector": inner}
    if isinstance(bijector, tfb.Chain):
        bijectors = [
            _serialize_bijector(b, arrays, f"{prefix}_{i}")
            for i, b in enumerate(bijector.bijectors)
        ]
        return {"kind": "Chain", "bijectors": bijectors}
    raise ValueError("Unsupported bijector for safe serialization.")


def _deserialize_bijector(spec, arrays):
    kind = spec["kind"]
    if kind == "Identity":
        return tfb.Identity()
    if kind == "Shift":
        shift = _resolve_ref(spec["kwargs"]["shift"], arrays)
        return tfb.Shift(shift=shift)
    if kind == "Scale":
        scale = _resolve_ref(spec["kwargs"]["scale"], arrays)
        return tfb.Scale(scale=scale)
    if kind == "Sigmoid":
        kwargs = spec.get("kwargs", {})
        if kwargs:
            low = _resolve_ref(kwargs.get("low"), arrays)
            high = _resolve_ref(kwargs.get("high"), arrays)
            return tfb.Sigmoid(low=low, high=high)
        return tfb.Sigmoid()
    if kind == "Softplus":
        kwargs = spec.get("kwargs", {})
        if kwargs:
            hinge_softness = _resolve_ref(
                kwargs.get("hinge_softness"), arrays
            )
            return tfb.Softplus(hinge_softness=hinge_softness)
        return tfb.Softplus()
    if kind == "Exp":
        return tfb.Exp()
    if kind == "SoftmaxCentered":
        return tfb.SoftmaxCentered()
    if kind == "Invert":
        bijector = _deserialize_bijector(spec["bijector"], arrays)
        return tfb.Invert(bijector)
    if kind == "Chain":
        bijectors = [
            _deserialize_bijector(b, arrays) for b in spec["bijectors"]
        ]
        return tfb.Chain(bijectors)
    raise ValueError(f"Unknown bijector kind '{kind}'.")


def _serialize_distribution(density, arrays):
    metadata = density.metadata or {}
    distribution = metadata.get("distribution")
    if distribution is None:
        raise ValueError(
            "Density serialization requires a distribution metadata entry."
        )
    if isinstance(distribution, tfd.MultivariateNormalDiag):
        arrays["loc"] = distribution.loc
        arrays["scale_diag"] = _get_scale_diag(distribution)
        return {
            "density_kind": "gaussian",
            "covariance": "diag",
            "loc": _ref("loc"),
            "scale_diag": _ref("scale_diag"),
        }
    if isinstance(distribution, tfd.MultivariateNormalFullCovariance):
        arrays["loc"] = distribution.loc
        arrays["covariance"] = distribution.covariance()
        return {
            "density_kind": "gaussian",
            "covariance": "full",
            "loc": _ref("loc"),
            "covariance": _ref("covariance"),
        }
    if isinstance(distribution, tfd.MixtureSameFamily):
        mixture = distribution.mixture_distribution
        components = distribution.components_distribution
        arrays["weights"] = _get_probs(mixture)
        arrays["means"] = components.loc
        if isinstance(components, tfd.MultivariateNormalDiag):
            arrays["scale_diag"] = _get_scale_diag(components)
            covariance = "diag"
        elif isinstance(components, tfd.MultivariateNormalFullCovariance):
            arrays["covariance"] = components.covariance()
            covariance = "full"
        else:
            raise ValueError("Unsupported GMM component distribution.")
        spec = {
            "density_kind": "gmm",
            "covariance": covariance,
            "weights": _ref("weights"),
            "means": _ref("means"),
        }
        if covariance == "diag":
            spec["scale_diag"] = _ref("scale_diag")
        else:
            spec["covariance"] = _ref("covariance")
        return spec
    raise ValueError("Unsupported density distribution.")


def _deserialize_distribution(spec, arrays):
    kind = spec["density_kind"]
    covariance = spec["covariance"]
    if kind == "gaussian" and covariance == "diag":
        loc = _resolve_ref(spec["loc"], arrays)
        scale = _resolve_ref(spec["scale_diag"], arrays)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
    if kind == "gaussian" and covariance == "full":
        loc = _resolve_ref(spec["loc"], arrays)
        cov = _resolve_ref(spec["covariance"], arrays)
        return tfd.MultivariateNormalFullCovariance(
            loc=loc, covariance_matrix=cov
        )
    if kind == "gmm" and covariance == "diag":
        weights = _resolve_ref(spec["weights"], arrays)
        means = _resolve_ref(spec["means"], arrays)
        scale = _resolve_ref(spec["scale_diag"], arrays)
        components = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=scale
        )
        return tfd.MixtureSameFamily(
            tfd.Categorical(probs=weights), components
        )
    if kind == "gmm" and covariance == "full":
        weights = _resolve_ref(spec["weights"], arrays)
        means = _resolve_ref(spec["means"], arrays)
        cov = _resolve_ref(spec["covariance"], arrays)
        components = tfd.MultivariateNormalFullCovariance(
            loc=means, covariance_matrix=cov
        )
        return tfd.MixtureSameFamily(
            tfd.Categorical(probs=weights), components
        )
    raise ValueError("Unsupported distribution specification.")


def _serialize_distribution_obj(distribution, arrays, prefix):
    if isinstance(distribution, tfd.Normal):
        arrays[f"{prefix}_loc"] = distribution.loc
        arrays[f"{prefix}_scale"] = distribution.scale
        return {
            "kind": "Normal",
            "loc": _ref(f"{prefix}_loc"),
            "scale": _ref(f"{prefix}_scale"),
        }
    if isinstance(distribution, tfd.Deterministic):
        arrays[f"{prefix}_loc"] = distribution.loc
        return {
            "kind": "Deterministic",
            "loc": _ref(f"{prefix}_loc"),
        }
    if isinstance(distribution, tfd.Laplace):
        arrays[f"{prefix}_loc"] = distribution.loc
        arrays[f"{prefix}_scale"] = distribution.scale
        return {
            "kind": "Laplace",
            "loc": _ref(f"{prefix}_loc"),
            "scale": _ref(f"{prefix}_scale"),
        }
    if isinstance(distribution, tfd.StudentT):
        arrays[f"{prefix}_df"] = distribution.df
        arrays[f"{prefix}_loc"] = distribution.loc
        arrays[f"{prefix}_scale"] = distribution.scale
        return {
            "kind": "StudentT",
            "df": _ref(f"{prefix}_df"),
            "loc": _ref(f"{prefix}_loc"),
            "scale": _ref(f"{prefix}_scale"),
        }
    if isinstance(distribution, tfd.LogNormal):
        arrays[f"{prefix}_loc"] = distribution.loc
        arrays[f"{prefix}_scale"] = distribution.scale
        return {
            "kind": "LogNormal",
            "loc": _ref(f"{prefix}_loc"),
            "scale": _ref(f"{prefix}_scale"),
        }
    if isinstance(distribution, tfd.MixtureSameFamily):
        arrays[f"{prefix}_weights"] = _get_probs(
            distribution.mixture_distribution
        )
        components = _serialize_distribution_obj(
            distribution.components_distribution,
            arrays,
            f"{prefix}_components",
        )
        return {
            "kind": "MixtureSameFamily",
            "weights": _ref(f"{prefix}_weights"),
            "components": components,
        }
    if isinstance(distribution, tfd.Uniform):
        arrays[f"{prefix}_low"] = distribution.low
        arrays[f"{prefix}_high"] = distribution.high
        return {
            "kind": "Uniform",
            "low": _ref(f"{prefix}_low"),
            "high": _ref(f"{prefix}_high"),
        }
    if isinstance(distribution, tfd.Beta):
        arrays[f"{prefix}_a"] = distribution.concentration1
        arrays[f"{prefix}_b"] = distribution.concentration0
        return {
            "kind": "Beta",
            "concentration1": _ref(f"{prefix}_a"),
            "concentration0": _ref(f"{prefix}_b"),
        }
    if isinstance(distribution, tfd.Poisson):
        spec = {"kind": "Poisson"}
        rate = getattr(distribution, "rate", None)
        if rate is not None:
            arrays[f"{prefix}_rate"] = rate
            spec["rate"] = _ref(f"{prefix}_rate")
            return spec
        log_rate = getattr(distribution, "log_rate", None)
        arrays[f"{prefix}_log_rate"] = log_rate
        spec["log_rate"] = _ref(f"{prefix}_log_rate")
        return spec
    if isinstance(distribution, tfd.Bernoulli):
        spec = {"kind": "Bernoulli", "dtype": _dtype_name(distribution.dtype)}
        probs = getattr(distribution, "probs", None)
        logits = getattr(distribution, "logits", None)
        if probs is not None:
            arrays[f"{prefix}_probs"] = probs
            spec["probs"] = _ref(f"{prefix}_probs")
            return spec
        arrays[f"{prefix}_logits"] = logits
        spec["logits"] = _ref(f"{prefix}_logits")
        return spec
    if isinstance(distribution, tfd.Categorical):
        spec = {
            "kind": "Categorical",
            "dtype": _dtype_name(distribution.dtype),
        }
        probs = getattr(distribution, "probs", None)
        logits = getattr(distribution, "logits", None)
        if probs is not None:
            arrays[f"{prefix}_probs"] = probs
            spec["probs"] = _ref(f"{prefix}_probs")
            return spec
        arrays[f"{prefix}_logits"] = logits
        spec["logits"] = _ref(f"{prefix}_logits")
        return spec
    if isinstance(distribution, tfd.RelaxedOneHotCategorical):
        spec = {"kind": "RelaxedOneHotCategorical"}
        arrays[f"{prefix}_temperature"] = distribution.temperature
        spec["temperature"] = _ref(f"{prefix}_temperature")
        probs = getattr(distribution, "probs", None)
        logits = getattr(distribution, "logits", None)
        if probs is not None:
            arrays[f"{prefix}_probs"] = probs
            spec["probs"] = _ref(f"{prefix}_probs")
            return spec
        arrays[f"{prefix}_logits"] = logits
        spec["logits"] = _ref(f"{prefix}_logits")
        return spec
    if isinstance(distribution, tfd.VonMises):
        arrays[f"{prefix}_loc"] = distribution.loc
        arrays[f"{prefix}_concentration"] = distribution.concentration
        return {
            "kind": "VonMises",
            "loc": _ref(f"{prefix}_loc"),
            "concentration": _ref(f"{prefix}_concentration"),
        }
    if isinstance(distribution, tfd.TruncatedNormal):
        arrays[f"{prefix}_loc"] = distribution.loc
        arrays[f"{prefix}_scale"] = distribution.scale
        arrays[f"{prefix}_low"] = distribution.low
        arrays[f"{prefix}_high"] = distribution.high
        return {
            "kind": "TruncatedNormal",
            "loc": _ref(f"{prefix}_loc"),
            "scale": _ref(f"{prefix}_scale"),
            "low": _ref(f"{prefix}_low"),
            "high": _ref(f"{prefix}_high"),
        }
    if isinstance(distribution, tfd.TransformedDistribution):
        base = _serialize_distribution_obj(
            distribution.distribution, arrays, f"{prefix}_base"
        )
        bijector = _serialize_bijector(
            distribution.bijector, arrays, f"{prefix}_bijector"
        )
        return {
            "kind": "TransformedDistribution",
            "base": base,
            "bijector": bijector,
        }
    if isinstance(distribution, tfd.QuantizedDistribution):
        base = _serialize_distribution_obj(
            distribution.distribution, arrays, f"{prefix}_base"
        )
        spec = {"kind": "QuantizedDistribution", "base": base}
        if getattr(distribution, "low", None) is not None:
            arrays[f"{prefix}_low"] = distribution.low
            spec["low"] = _ref(f"{prefix}_low")
        if getattr(distribution, "high", None) is not None:
            arrays[f"{prefix}_high"] = distribution.high
            spec["high"] = _ref(f"{prefix}_high")
        return spec
    if isinstance(distribution, tfd.Independent):
        base = _serialize_distribution_obj(
            distribution.distribution, arrays, f"{prefix}_base"
        )
        return {
            "kind": "Independent",
            "base": base,
            "reinterpreted_batch_ndims": int(
                distribution.reinterpreted_batch_ndims
            ),
        }
    raise ValueError(
        "Unsupported prior distribution for safe serialization. "
        "Use a supported distribution type."
    )


def _deserialize_distribution_obj(spec, arrays):
    kind = spec["kind"]
    if kind == "Normal":
        loc = _resolve_ref(spec["loc"], arrays)
        scale = _resolve_ref(spec["scale"], arrays)
        return tfd.Normal(loc=loc, scale=scale)
    if kind == "Deterministic":
        loc = _resolve_ref(spec["loc"], arrays)
        return tfd.Deterministic(loc=loc)
    if kind == "Laplace":
        loc = _resolve_ref(spec["loc"], arrays)
        scale = _resolve_ref(spec["scale"], arrays)
        return tfd.Laplace(loc=loc, scale=scale)
    if kind == "StudentT":
        df = _resolve_ref(spec["df"], arrays)
        loc = _resolve_ref(spec["loc"], arrays)
        scale = _resolve_ref(spec["scale"], arrays)
        return tfd.StudentT(df=df, loc=loc, scale=scale)
    if kind == "LogNormal":
        loc = _resolve_ref(spec["loc"], arrays)
        scale = _resolve_ref(spec["scale"], arrays)
        return tfd.LogNormal(loc=loc, scale=scale)
    if kind == "MixtureSameFamily":
        weights = _resolve_ref(spec["weights"], arrays)
        components = _deserialize_distribution_obj(spec["components"], arrays)
        return tfd.MixtureSameFamily(
            tfd.Categorical(probs=weights), components
        )
    if kind == "Uniform":
        low = _resolve_ref(spec["low"], arrays)
        high = _resolve_ref(spec["high"], arrays)
        return tfd.Uniform(low=low, high=high)
    if kind == "Beta":
        a = _resolve_ref(spec["concentration1"], arrays)
        b = _resolve_ref(spec["concentration0"], arrays)
        return tfd.Beta(concentration1=a, concentration0=b)
    if kind == "Poisson":
        if "rate" in spec:
            rate = _resolve_ref(spec["rate"], arrays)
            return tfd.Poisson(rate=rate)
        log_rate = _resolve_ref(spec["log_rate"], arrays)
        return tfd.Poisson(log_rate=log_rate)
    if kind == "Bernoulli":
        dtype = jp.dtype(spec["dtype"])
        if "probs" in spec:
            probs = _resolve_ref(spec["probs"], arrays)
            return tfd.Bernoulli(probs=probs, dtype=dtype)
        logits = _resolve_ref(spec["logits"], arrays)
        return tfd.Bernoulli(logits=logits, dtype=dtype)
    if kind == "Categorical":
        dtype = jp.dtype(spec["dtype"])
        if "probs" in spec:
            probs = _resolve_ref(spec["probs"], arrays)
            return tfd.Categorical(probs=probs, dtype=dtype)
        logits = _resolve_ref(spec["logits"], arrays)
        return tfd.Categorical(logits=logits, dtype=dtype)
    if kind == "RelaxedOneHotCategorical":
        temperature = _resolve_ref(spec["temperature"], arrays)
        if "probs" in spec:
            probs = _resolve_ref(spec["probs"], arrays)
            return tfd.RelaxedOneHotCategorical(temperature, probs=probs)
        logits = _resolve_ref(spec["logits"], arrays)
        return tfd.RelaxedOneHotCategorical(temperature, logits=logits)
    if kind == "VonMises":
        loc = _resolve_ref(spec["loc"], arrays)
        concentration = _resolve_ref(spec["concentration"], arrays)
        return tfd.VonMises(loc=loc, concentration=concentration)
    if kind == "TruncatedNormal":
        loc = _resolve_ref(spec["loc"], arrays)
        scale = _resolve_ref(spec["scale"], arrays)
        low = _resolve_ref(spec["low"], arrays)
        high = _resolve_ref(spec["high"], arrays)
        return tfd.TruncatedNormal(
            loc=loc, scale=scale, low=low, high=high
        )
    if kind == "TransformedDistribution":
        base = _deserialize_distribution_obj(spec["base"], arrays)
        bijector = _deserialize_bijector(spec["bijector"], arrays)
        return tfd.TransformedDistribution(base, bijector)
    if kind == "QuantizedDistribution":
        base = _deserialize_distribution_obj(spec["base"], arrays)
        kwargs = {}
        if "low" in spec:
            kwargs["low"] = _resolve_ref(spec["low"], arrays)
        if "high" in spec:
            kwargs["high"] = _resolve_ref(spec["high"], arrays)
        return tfd.QuantizedDistribution(distribution=base, **kwargs)
    if kind == "Independent":
        base = _deserialize_distribution_obj(spec["base"], arrays)
        return tfd.Independent(
            base, reinterpreted_batch_ndims=spec["reinterpreted_batch_ndims"]
        )
    raise ValueError("Unsupported distribution specification.")


def _sample_spec(sample):
    samples, kind = _samples_to_dict(sample)
    shapes = {}
    dtypes = {}
    for name, value in samples.items():
        array = jp.asarray(value)
        shapes[name] = list(array.shape)
        dtypes[name] = str(array.dtype)
    return {
        "kind": kind,
        "names": list(samples.keys()),
        "shapes": shapes,
        "dtypes": dtypes,
    }


def _dummy_sample(spec, latent_space):
    values = {}
    for name in spec["names"]:
        shape = tuple(spec["shapes"][name])
        dtype = spec["dtypes"][name]
        values[name] = jp.zeros(shape, dtype=dtype)
    if spec["kind"] == "namedtuple":
        return latent_space.Sample(**values)
    if spec["kind"] == "dict":
        return values
    return values["value"]


def _clean_config(config):
    if config is None:
        return {}
    allowed = [
        "method",
        "num_samples",
        "num_chains",
        "sigma",
        "warmup",
        "space",
        "progress",
        "tune",
    ]
    return {key: _jsonify(config[key]) for key in allowed if key in config}


def _samples_to_dict(samples):
    if hasattr(samples, "_asdict"):
        return samples._asdict(), "namedtuple"
    if isinstance(samples, dict):
        return samples, "dict"
    return {"value": samples}, "array"


def _dict_to_samples(samples, kind, latent_space):
    if kind == "namedtuple":
        return latent_space.Sample(**samples)
    if kind == "dict":
        return samples
    return samples["value"]


def _encode_kwargs(kwargs, arrays, prefix):
    encoded = {}
    for key, value in kwargs.items():
        encoded[key] = _encode_value(value, arrays, f"{prefix}_{key}")
    return encoded


def _decode_kwargs(kwargs, arrays):
    decoded = {}
    for key, value in kwargs.items():
        decoded[key] = _decode_value(value, arrays)
    return decoded


def _encode_value(value, arrays, name):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "shape"):
        arrays[name] = value
        return _ref(name)
    if isinstance(value, (list, tuple)):
        arrays[name] = jp.asarray(value)
        return _ref(name)
    return str(value)


def _decode_value(value, arrays):
    if isinstance(value, str) and value.startswith("arrays:"):
        return _resolve_ref(value, arrays)
    return value


def _get_probs(mixture):
    if hasattr(mixture, "probs_parameter"):
        return mixture.probs_parameter()
    if hasattr(mixture, "probs"):
        return mixture.probs
    if hasattr(mixture, "logits"):
        return jax.nn.softmax(mixture.logits)
    raise ValueError("Unable to extract mixture probabilities.")


def _get_scale_diag(distribution):
    scale_diag = getattr(distribution, "scale_diag", None)
    if scale_diag is not None:
        return scale_diag
    scale = getattr(distribution, "scale", None)
    if scale is None:
        raise ValueError("Unable to extract scale_diag.")
    if hasattr(scale, "diag_part"):
        return scale.diag_part()
    if hasattr(scale, "diagonal"):
        return scale.diagonal()
    raise ValueError("Unable to extract scale_diag.")


def _dtype_name(dtype):
    return jp.dtype(dtype).name


def _ref(name):
    return f"arrays:{name}"


def _resolve_ref(ref, arrays):
    if isinstance(ref, str) and ref.startswith("arrays:"):
        return arrays[ref.split(":", 1)[1]]
    return ref


def _jsonify(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
