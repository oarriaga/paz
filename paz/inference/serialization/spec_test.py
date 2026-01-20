import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior
from paz.inference.serialization.spec import (
    _build_latent_space,
    _clean_config,
    _decode_kwargs,
    _decode_value,
    _deserialize_bijector,
    _deserialize_distribution,
    _deserialize_distribution_obj,
    _dict_to_samples,
    _dtype_name,
    _dummy_sample,
    _encode_kwargs,
    _encode_value,
    _get_probs,
    _get_scale_diag,
    _jsonify,
    _latent_space_spec,
    _ref,
    _resolve_ref,
    _sample_spec,
    _samples_to_dict,
    _serialize_bijector,
    _serialize_distribution,
    _serialize_distribution_obj,
 )


tfd = tfp.distributions
tfb = tfp.bijectors


class _DummyDensity:
    def __init__(self, distribution):
        self.metadata = {"distribution": distribution}


def test_ref_and_resolve_ref():
    arrays = {"x": jp.array([1.0])}
    ref = _ref("x")
    assert ref == "arrays:x"
    assert jp.allclose(_resolve_ref(ref, arrays), arrays["x"])
    assert _resolve_ref(3, arrays) == 3


def test_dtype_name():
    assert _dtype_name(jp.float32) == "float32"


def test_jsonify_values():
    assert _jsonify("x") == "x"
    assert _jsonify(1) == 1
    assert _jsonify(jp.array([1, 2])) == [1, 2]


def test_encode_decode_value():
    arrays = {}
    encoded = _encode_value(jp.array([1.0]), arrays, "arr")
    decoded = _decode_value(encoded, arrays)
    assert jp.allclose(decoded, jp.array([1.0]))


def test_encode_decode_kwargs():
    arrays = {}
    kwargs = {"x": jp.array([1.0]), "y": 2}
    encoded = _encode_kwargs(kwargs, arrays, "test")
    decoded = _decode_kwargs(encoded, arrays)
    assert jp.allclose(decoded["x"], jp.array([1.0]))
    assert decoded["y"] == 2


def test_samples_to_dict_roundtrip():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    latent_space = build_latent_space([prior])
    sample = latent_space.Sample(x=jp.array([0.2]))
    samples, kind = _samples_to_dict(sample)
    restored = _dict_to_samples(samples, kind, latent_space)
    assert restored == sample


def test_samples_to_dict_dict_kind():
    samples, kind = _samples_to_dict({"x": 1})
    assert kind == "dict"
    assert _dict_to_samples(samples, kind, None) == {"x": 1}


def test_sample_spec_and_dummy_sample():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    latent_space = build_latent_space([prior])
    sample = latent_space.Sample(x=jp.array([0.2, 0.3]))
    spec = _sample_spec(sample)
    dummy = _dummy_sample(spec, latent_space)
    assert dummy.x.shape == (2,)
    assert str(dummy.x.dtype) == str(jp.array(sample.x).dtype)


def test_clean_config_filters_keys():
    config = {"method": "mh", "sigma": 0.1, "extra": 3}
    cleaned = _clean_config(config)
    assert cleaned == {"method": "mh", "sigma": 0.1}


def test_latent_space_spec_roundtrip():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x", bijector=tfb.Exp())
    latent_space = build_latent_space([prior])
    arrays = {}
    spec = _latent_space_spec(latent_space, arrays, "latent")
    rebuilt = _build_latent_space(spec, arrays)
    assert rebuilt.names == latent_space.names
    assert isinstance(rebuilt.bijectors["x"], tfb.Exp)


def test_bijector_roundtrip():
    bijectors = [
        tfb.Identity(),
        tfb.Shift(1.0),
        tfb.Scale(2.0),
        tfb.Sigmoid(low=-1.0, high=1.0),
        tfb.Softplus(hinge_softness=0.3),
        tfb.Exp(),
        tfb.SoftmaxCentered(),
        tfb.Invert(tfb.Scale(3.0)),
        tfb.Chain([tfb.Shift(1.0), tfb.Scale(2.0)]),
    ]
    for bijector in bijectors:
        arrays = {}
        spec = _serialize_bijector(bijector, arrays, "b")
        restored = _deserialize_bijector(spec, arrays)
        assert isinstance(restored, type(bijector))


def test_get_probs_from_logits():
    logits = jp.array([0.1, 0.2, 0.3])
    probs = _get_probs(tfd.Categorical(logits=logits))
    assert jp.allclose(probs, jax.nn.softmax(logits))


def test_get_scale_diag():
    class DummyScaleDiag:
        def __init__(self, scale_diag):
            self.scale_diag = scale_diag

    class DummyScale:
        def __init__(self, scale):
            self.scale = scale

    diag = jp.array([1.0, 2.0])
    assert jp.allclose(_get_scale_diag(DummyScaleDiag(diag)), diag)
    assert jp.allclose(_get_scale_diag(DummyScale(jp.eye(2))), jp.array([1.0, 1.0]))


def test_serialize_deserialize_density_distribution():
    arrays = {}
    dist = tfd.MultivariateNormalDiag(loc=jp.zeros(2), scale_diag=jp.ones(2))
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    assert isinstance(restored, tfd.MultivariateNormalDiag)


def test_serialize_deserialize_gmm_distribution():
    arrays = {}
    weights = jp.array([0.3, 0.7])
    means = jp.array([[0.0, 0.0], [1.0, 1.0]])
    scales = jp.array([[1.0, 1.0], [0.5, 0.5]])
    components = tfd.MultivariateNormalDiag(loc=means, scale_diag=scales)
    dist = tfd.MixtureSameFamily(tfd.Categorical(probs=weights), components)
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    assert isinstance(restored, tfd.MixtureSameFamily)


def test_serialize_deserialize_distribution_obj_roundtrip():
    cases = [
        (tfd.Normal(0.0, 1.0), jp.array(0.2)),
        (tfd.Deterministic(0.25), jp.array(0.25)),
        (tfd.Laplace(0.0, 0.5), jp.array(0.2)),
        (tfd.StudentT(5.0, 0.0, 0.7), jp.array(0.1)),
        (tfd.LogNormal(0.0, 0.4), jp.array(1.2)),
        (tfd.Uniform(-1.0, 1.0), jp.array(0.2)),
        (tfd.Beta(2.0, 3.0), jp.array(0.4)),
        (tfd.Poisson(rate=3.0), jp.array(2.0)),
        (tfd.Poisson(log_rate=jp.log(3.0)), jp.array(2.0)),
        (tfd.Bernoulli(probs=0.7), jp.array(1.0)),
        (tfd.Bernoulli(logits=jp.array(0.2)), jp.array(0.0)),
        (tfd.Categorical(probs=jp.array([0.2, 0.8])), jp.array(1.0)),
        (tfd.Categorical(logits=jp.array([0.1, 0.2])), jp.array(0.0)),
        (
            tfd.RelaxedOneHotCategorical(
                0.5, probs=jp.array([0.2, 0.3, 0.5])
            ),
            jp.array([0.8, 0.1, 0.1]),
        ),
        (
            tfd.RelaxedOneHotCategorical(
                0.5, logits=jp.array([0.1, 0.2, 0.3])
            ),
            jp.array([0.1, 0.2, 0.7]),
        ),
        (tfd.VonMises(0.0, 2.0), jp.array(0.1)),
        (tfd.TruncatedNormal(0.0, 1.0, -1.0, 1.0), jp.array(0.2)),
        (
            tfd.TransformedDistribution(tfd.Normal(0.0, 1.0), tfb.Exp()),
            jp.array(1.1),
        ),
        (
            tfd.QuantizedDistribution(
                tfd.Normal(0.0, 1.0), low=-2.0, high=2.0
            ),
            jp.array(0.0),
        ),
        (
            tfd.Independent(
                tfd.Normal(jp.zeros(2), 1.0), reinterpreted_batch_ndims=1
            ),
            jp.array([0.1, -0.2]),
        ),
    ]
    for distribution, value in cases:
        arrays = {}
        spec = _serialize_distribution_obj(distribution, arrays, "dist")
        restored = _deserialize_distribution_obj(spec, arrays)
        log_prob = distribution.log_prob(value)
        log_prob_restored = restored.log_prob(value)
        assert jp.allclose(log_prob, log_prob_restored)
