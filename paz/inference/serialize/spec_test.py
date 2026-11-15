import jax
import jax.numpy as jp

from paz.backend import distributions
from paz.backend import bijectors
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
    _get_probs,
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
    prior = Prior(distributions.Normal(0.0, 1.0), name="x")
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
    prior = Prior(distributions.Normal(0.0, 1.0), name="x")
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
    normal = distributions.Normal(0.0, 1.0)
    prior = Prior(normal, name="x", bijector=bijectors.Exp())
    latent_space = build_latent_space([prior])
    arrays = {}
    spec = _latent_space_spec(latent_space, arrays, "latent")
    rebuilt = _build_latent_space(spec, arrays)
    assert rebuilt.names == latent_space.names
    assert isinstance(rebuilt.bijectors["x"], bijectors.Exp)


def test_bijector_roundtrip():
    cases = [
        bijectors.Identity(),
        bijectors.Shift(1.0),
        bijectors.Scale(2.0),
        bijectors.Sigmoid(low=-1.0, high=1.0),
        bijectors.Softplus(hinge_softness=0.3),
        bijectors.Exp(),
        bijectors.SoftmaxCentered(),
        bijectors.Invert(bijectors.Scale(3.0)),
        bijectors.Chain([bijectors.Shift(1.0), bijectors.Scale(2.0)]),
    ]
    for bijector in cases:
        arrays = {}
        spec = _serialize_bijector(bijector, arrays, "b")
        restored = _deserialize_bijector(spec, arrays)
        assert isinstance(restored, type(bijector))


def test_get_probs_from_logits():
    logits = jp.array([0.1, 0.2, 0.3])
    probs = _get_probs(distributions.Categorical(logits=logits))
    assert jp.allclose(probs, jax.nn.softmax(logits))


def test_serialize_deserialize_density_distribution():
    arrays = {}
    dist = distributions.MultivariateNormalDiag(jp.zeros(2), jp.ones(2))
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    assert isinstance(restored, distributions.MultivariateNormalDiag)


def test_serialize_deserialize_gmm_distribution():
    arrays = {}
    weights = jp.array([0.3, 0.7])
    means = jp.array([[0.0, 0.0], [1.0, 1.0]])
    scales = jp.array([[1.0, 1.0], [0.5, 0.5]])
    components = distributions.MultivariateNormalDiag(means, scales)
    mixture = distributions.Categorical(probs=weights)
    dist = distributions.MixtureSameFamily(mixture, components)
    spec = _serialize_distribution(_DummyDensity(dist), arrays)
    restored = _deserialize_distribution(spec, arrays)
    assert isinstance(restored, distributions.MixtureSameFamily)


def test_serialize_deserialize_distribution_obj_roundtrip():
    normal = distributions.Normal(0.0, 1.0)
    vector_normal = distributions.Normal(jp.zeros(2), 1.0)
    independent = distributions.Independent(vector_normal, 1)
    cases = [
        (normal, jp.array(0.2)),
        (distributions.Deterministic(0.25), jp.array(0.25)),
        (distributions.Laplace(0.0, 0.5), jp.array(0.2)),
        (distributions.StudentT(5.0, 0.0, 0.7), jp.array(0.1)),
        (distributions.LogNormal(0.0, 0.4), jp.array(1.2)),
        (distributions.Uniform(-1.0, 1.0), jp.array(0.2)),
        (distributions.Beta(2.0, 3.0), jp.array(0.4)),
        (distributions.Poisson(rate=3.0), jp.array(2.0)),
        (distributions.Poisson(log_rate=jp.log(3.0)), jp.array(2.0)),
        (distributions.Bernoulli(probs=0.7), jp.array(1.0)),
        (distributions.Bernoulli(logits=jp.array(0.2)), jp.array(0.0)),
        (distributions.Categorical(probs=jp.array([0.2, 0.8])), jp.array(1.0)),
        (distributions.Categorical(logits=jp.array([0.1, 0.2])), jp.array(0.0)),
        (
            distributions.RelaxedOneHotCategorical(
                0.5, probs=jp.array([0.2, 0.3, 0.5])
            ),
            jp.array([0.8, 0.1, 0.1]),
        ),
        (
            distributions.RelaxedOneHotCategorical(
                0.5, logits=jp.array([0.1, 0.2, 0.3])
            ),
            jp.array([0.1, 0.2, 0.7]),
        ),
        (distributions.VonMises(0.0, 2.0), jp.array(0.1)),
        (distributions.TruncatedNormal(0.0, 1.0, -1.0, 1.0), jp.array(0.2)),
        (
            distributions.TransformedDistribution(normal, bijectors.Exp()),
            jp.array(1.1),
        ),
        (
            distributions.QuantizedDistribution(
                distributions.Normal(0.0, 1.0), low=-2.0, high=2.0
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
