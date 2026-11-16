import jax.numpy as jp

import paz
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior
from paz.inference.serialize.spec import (
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
    prior = Prior(paz.distributions.Normal(0.0, 1.0), name="x")
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
    prior = Prior(paz.distributions.Normal(0.0, 1.0), name="x")
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
    normal = paz.distributions.Normal(0.0, 1.0)
    prior = Prior(normal, name="x", bijector=paz.bijectors.Exp())
    latent_space = build_latent_space([prior])
    arrays = {}
    spec = _latent_space_spec(latent_space, arrays, "latent")
    rebuilt = _build_latent_space(spec, arrays)
    assert rebuilt.names == latent_space.names
    assert isinstance(rebuilt.bijectors["x"], paz.bijectors.Exp)


def test_bijector_roundtrip():
    affine = [paz.bijectors.Shift(1.0), paz.bijectors.Scale(2.0)]
    cases = [
        paz.bijectors.Identity(),
        paz.bijectors.Shift(1.0),
        paz.bijectors.Scale(2.0),
        paz.bijectors.Sigmoid(low=-1.0, high=1.0),
        paz.bijectors.Softplus(hinge_softness=0.3),
        paz.bijectors.Exp(),
        paz.bijectors.SoftmaxCentered(),
        paz.bijectors.Invert(paz.bijectors.Scale(3.0)),
        paz.bijectors.Chain(affine),
    ]
    for bijector in cases:
        arrays = {}
        spec = _serialize_bijector(bijector, arrays, "b")
        restored = _deserialize_bijector(spec, arrays)
        assert isinstance(restored, type(bijector))


def test_serialize_deserialize_density_distribution():
    arrays = {}
    dist = paz.distributions.MultivariateNormalDiag(jp.zeros(2), jp.ones(2))
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    assert isinstance(restored, paz.distributions.MultivariateNormalDiag)


def test_serialize_deserialize_full_covariance_density():
    arrays = {}
    covariance = jp.array([[2.0, 0.3], [0.3, 1.0]])
    full = paz.distributions.MultivariateNormalFullCovariance
    dist = full(jp.array([0.5, -1.0]), covariance)
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    value = jp.array([0.1, 0.2])
    assert jp.allclose(restored.log_prob(value), dist.log_prob(value))


def test_serialize_deserialize_full_covariance_gmm():
    arrays = {}
    means = jp.array([[0.0, 0.0], [1.0, 1.0]])
    covariances = jp.stack([jp.eye(2), jp.array([[2.0, 0.3], [0.3, 1.0]])])
    full = paz.distributions.MultivariateNormalFullCovariance
    mixture = paz.distributions.Categorical(probs=jp.array([0.3, 0.7]))
    components = full(means, covariances)
    dist = paz.distributions.MixtureSameFamily(mixture, components)
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    value = jp.array([0.1, 0.2])
    assert jp.allclose(restored.log_prob(value), dist.log_prob(value))


def test_softplus_low_survives_roundtrip():
    arrays = {}
    bijector = paz.bijectors.Softplus(hinge_softness=0.5, low=2.0)
    spec = _serialize_bijector(bijector, arrays, "b")
    restored = _deserialize_bijector(spec, arrays)
    assert jp.allclose(restored(jp.array(0.3)), bijector(jp.array(0.3)))


def test_force_probs_survives_roundtrip():
    force = dict(force_probs_to_zero_outside_support=True)
    probs, logits = jp.array([0.2, 0.8]), jp.array([0.1, 0.2])
    cases = [
        (paz.distributions.Beta(2.0, 3.0, **force), 0.4),
        (paz.distributions.Poisson(rate=3.0, **force), 2.0),
        (paz.distributions.Poisson(log_rate=1.1, **force), 2.0),
        (paz.distributions.Categorical(probs=probs, **force), 1.0),
        (paz.distributions.Categorical(logits=logits, **force), 1.0),
    ]
    for distribution, value in cases:
        arrays = {}
        spec = _serialize_distribution_obj(distribution, arrays, "d")
        restored = _deserialize_distribution_obj(spec, arrays)
        assert restored.force_probs
        assert restored.dtype == distribution.dtype
        expected = distribution.log_prob(value)
        assert jp.allclose(restored.log_prob(value), expected)


def test_serialize_deserialize_gmm_distribution():
    arrays = {}
    weights = jp.array([0.3, 0.7])
    means = jp.array([[0.0, 0.0], [1.0, 1.0]])
    scales = jp.array([[1.0, 1.0], [0.5, 0.5]])
    components = paz.distributions.MultivariateNormalDiag(means, scales)
    mixture = paz.distributions.Categorical(probs=weights)
    dist = paz.distributions.MixtureSameFamily(mixture, components)
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    assert isinstance(restored, paz.distributions.MixtureSameFamily)


def test_serialize_deserialize_distribution_obj_roundtrip():
    normal = paz.distributions.Normal(0.0, 1.0)
    vector_normal = paz.distributions.Normal(jp.zeros(2), 1.0)
    independent = paz.distributions.Independent(vector_normal, 1)
    probs, logits = jp.array([0.2, 0.8]), jp.array([0.1, 0.2])
    categorical_probs = paz.distributions.Categorical(probs=probs)
    categorical_logits = paz.distributions.Categorical(logits=logits)
    exp = paz.bijectors.Exp()
    log_normal = paz.distributions.TransformedDistribution(normal, exp)
    cases = [
        (normal, jp.array(0.2)),
        (paz.distributions.Deterministic(0.25), jp.array(0.25)),
        (paz.distributions.Laplace(0.0, 0.5), jp.array(0.2)),
        (paz.distributions.StudentT(5.0, 0.0, 0.7), jp.array(0.1)),
        (paz.distributions.LogNormal(0.0, 0.4), jp.array(1.2)),
        (paz.distributions.Uniform(-1.0, 1.0), jp.array(0.2)),
        (paz.distributions.Beta(2.0, 3.0), jp.array(0.4)),
        (paz.distributions.Poisson(rate=3.0), jp.array(2.0)),
        (paz.distributions.Poisson(log_rate=jp.log(3.0)), jp.array(2.0)),
        (paz.distributions.Bernoulli(probs=0.7), jp.array(1.0)),
        (paz.distributions.Bernoulli(logits=jp.array(0.2)), jp.array(0.0)),
        (categorical_probs, jp.array(1.0)),
        (categorical_logits, jp.array(0.0)),
        (
            paz.distributions.RelaxedOneHotCategorical(
                0.5, probs=jp.array([0.2, 0.3, 0.5])
            ),
            jp.array([0.8, 0.1, 0.1]),
        ),
        (
            paz.distributions.RelaxedOneHotCategorical(
                0.5, logits=jp.array([0.1, 0.2, 0.3])
            ),
            jp.array([0.1, 0.2, 0.7]),
        ),
        (paz.distributions.VonMises(0.0, 2.0), jp.array(0.1)),
        (paz.distributions.TruncatedNormal(0.0, 1.0, -1.0, 1.0), jp.array(0.2)),
        (log_normal, jp.array(1.1)),
        (
            paz.distributions.QuantizedDistribution(
                paz.distributions.Normal(0.0, 1.0), low=-2.0, high=2.0
            ),
            jp.array(0.0),
        ),
        (independent, jp.array([0.1, -0.2])),
    ]
    for distribution, value in cases:
        arrays = {}
        spec = _serialize_distribution_obj(distribution, arrays, "dist")
        restored = _deserialize_distribution_obj(spec, arrays)
        log_prob = distribution.log_prob(value)
        log_prob_restored = restored.log_prob(value)
        assert jp.allclose(log_prob, log_prob_restored)
