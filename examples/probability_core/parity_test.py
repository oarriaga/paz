import pytest

pytest.skip(
    "Legacy parity suite replaced by value, gradient, and extreme tests.",
    allow_module_level=True,
)

import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from .bijectors import (
    Chain,
    Exp,
    Identity,
    Invert,
    Scale,
    Shift,
    Sigmoid,
    SoftmaxCentered,
    Softplus,
)
from .distributions import (
    Beta,
    Bernoulli,
    Categorical,
    Deterministic,
    Independent,
    Laplace,
    LogNormal,
    MixtureSameFamily,
    MultivariateNormalDiag,
    MultivariateNormalFullCovariance,
    Normal,
    Poisson,
    QuantizedDistribution,
    RelaxedOneHotCategorical,
    StudentT,
    TransformedDistribution,
    TruncatedNormal,
    Uniform,
    VonMises,
)

tfd = tfp.distributions
tfb = tfp.bijectors


def assert_close(ours, theirs, atol=1e-6, rtol=1e-6):
    ours = jp.asarray(ours)
    theirs = jp.asarray(theirs)
    assert ours.shape == theirs.shape
    assert ours.dtype == theirs.dtype
    assert jp.allclose(ours, theirs, atol=atol, rtol=rtol, equal_nan=True)


def assert_distribution_parity(ours, theirs, values, atol=1e-6, rtol=1e-6):
    assert tuple(ours.batch_shape) == tuple(theirs.batch_shape)
    assert tuple(ours.event_shape) == tuple(theirs.event_shape)
    assert ours.dtype == theirs.dtype

    our_log_prob = ours.log_prob(values)
    their_log_prob = theirs.log_prob(values)
    assert_close(our_log_prob, their_log_prob, atol=atol, rtol=rtol)

    our_prob = ours.prob(values)
    their_prob = theirs.prob(values)
    assert_close(our_prob, their_prob, atol=atol, rtol=rtol)


def assert_sample_parity(
    ours, theirs, num_samples, key, atol_mean=5e-2, atol_std=5e-2
):
    our_samples = jp.asarray(ours.sample(num_samples, seed=key))
    their_samples = jp.asarray(theirs.sample(num_samples, seed=key))
    assert our_samples.shape == their_samples.shape
    assert our_samples.dtype == their_samples.dtype

    our_mean = jp.mean(our_samples, axis=0)
    their_mean = jp.mean(their_samples, axis=0)
    assert_close(our_mean, their_mean, atol=atol_mean, rtol=0.0)

    our_std = jp.std(our_samples, axis=0)
    their_std = jp.std(their_samples, axis=0)
    assert_close(our_std, their_std, atol=atol_std, rtol=0.0)


def assert_method_parity(ours, theirs, method_name, *args, atol=1e-6, rtol=1e-6):
    our_value = getattr(ours, method_name)(*args)
    their_value = getattr(theirs, method_name)(*args)
    assert_close(our_value, their_value, atol=atol, rtol=rtol)


def assert_bijector_parity(
    ours, theirs, values, event_ndims=0, atol=1e-6, rtol=1e-6
):
    our_forward = ours(values)
    their_forward = theirs(values)
    assert_close(our_forward, their_forward, atol=atol, rtol=rtol)

    our_inverse = ours.inverse(their_forward)
    their_inverse = theirs.inverse(their_forward)
    assert_close(our_inverse, their_inverse, atol=atol, rtol=rtol)

    our_log_det = ours.forward_log_det_jacobian(values, event_ndims)
    their_log_det = theirs.forward_log_det_jacobian(values, event_ndims)
    assert_close(our_log_det, their_log_det, atol=atol, rtol=rtol)


def build_distribution_cases():
    bounded_bijector = tfb.Sigmoid(-2.0, 3.0)
    bounded_values = bounded_bijector(jp.array([-1.5, 0.0, 1.2]))
    return [
        pytest.param(
            Normal(0.5, 2.0),
            tfd.Normal(0.5, 2.0),
            jp.array([-1.5, -0.2, 0.3, 2.0]),
            id="normal_scalar",
        ),
        pytest.param(
            Normal(jp.array([0.0, 1.0]), jp.array([1.0, 2.0])),
            tfd.Normal(jp.array([0.0, 1.0]), jp.array([1.0, 2.0])),
            jp.array([[0.2, 1.3], [-0.5, 2.7]]),
            id="normal_broadcast",
        ),
        pytest.param(
            Uniform(0.1, 0.9),
            tfd.Uniform(0.1, 0.9),
            jp.array([-0.2, 0.1, 0.4, 0.9, 1.2]),
            id="uniform_scalar",
        ),
        pytest.param(
            Independent(
                Normal(jp.array([0.0, 1.0]), jp.array([1.0, 2.0])),
                1,
            ),
            tfd.Independent(
                tfd.Normal(jp.array([0.0, 1.0]), jp.array([1.0, 2.0])),
                1,
            ),
            jp.array([0.2, 1.3]),
            id="independent_1d",
        ),
        pytest.param(
            Independent(
                Normal(jp.zeros((2, 3)), jp.ones((2, 3))),
                2,
            ),
            tfd.Independent(
                tfd.Normal(jp.zeros((2, 3)), jp.ones((2, 3))),
                2,
            ),
            jp.array([[0.2, -0.3, 0.8], [1.1, -1.0, 0.4]]),
            id="independent_2d",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(0.0, 1.0),
                Chain([Shift(1.5), Scale(0.5)]),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(0.0, 1.0),
                tfb.Chain([tfb.Shift(1.5), tfb.Scale(0.5)]),
            ),
            jp.array([0.5, 1.0, 1.5]),
            id="transformed_affine",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(0.0, 1.0),
                Sigmoid(-2.0, 3.0),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(0.0, 1.0),
                tfb.Sigmoid(-2.0, 3.0),
            ),
            bounded_values,
            id="transformed_bounded",
        ),
    ]


def build_bijector_cases():
    simplex_values = jax.nn.softmax(jp.array([0.2, -0.4, 0.0, -0.3]))
    return [
        pytest.param(
            Identity(),
            tfb.Identity(),
            jp.array([-2.0, -0.1, 1.7]),
            0,
            id="identity",
        ),
        pytest.param(
            Shift(1.2),
            tfb.Shift(1.2),
            jp.array([-2.0, -0.1, 1.7]),
            0,
            id="shift",
        ),
        pytest.param(
            Scale(-0.7),
            tfb.Scale(-0.7),
            jp.array([-2.0, -0.1, 1.7]),
            0,
            id="scale",
        ),
        pytest.param(
            Sigmoid(),
            tfb.Sigmoid(),
            jp.array([-2.0, -0.1, 1.7]),
            0,
            id="sigmoid",
        ),
        pytest.param(
            Sigmoid(-2.0, 3.0),
            tfb.Sigmoid(-2.0, 3.0),
            jp.array([-2.0, -0.1, 1.7]),
            0,
            id="sigmoid_bounded",
        ),
        pytest.param(
            Chain([Shift(1.0), Scale(0.5), Sigmoid()]),
            tfb.Chain([tfb.Shift(1.0), tfb.Scale(0.5), tfb.Sigmoid()]),
            jp.array([-2.0, -0.1, 1.7]),
            0,
            id="chain_affine_sigmoid",
        ),
        pytest.param(
            Chain([Shift(-1.0), Scale(2.5)]),
            tfb.Chain([tfb.Shift(-1.0), tfb.Scale(2.5)]),
            jp.array([-1.5, 0.0, 2.0]),
            0,
            id="chain_affine",
        ),
        pytest.param(
            Exp(),
            tfb.Exp(),
            jp.array([-1.0, 0.2, 1.5]),
            0,
            id="exp",
        ),
        pytest.param(
            Softplus(),
            tfb.Softplus(),
            jp.array([-1.0, 0.2, 1.5]),
            0,
            id="softplus",
        ),
        pytest.param(
            Softplus(hinge_softness=0.3, low=0.2),
            tfb.Softplus(hinge_softness=0.3, low=0.2),
            jp.array([-1.0, 0.2, 1.5]),
            0,
            id="softplus_shifted",
        ),
        pytest.param(
            Invert(Scale(3.0)),
            tfb.Invert(tfb.Scale(3.0)),
            jp.array([-1.5, 0.0, 2.0]),
            0,
            id="invert_scale",
        ),
        pytest.param(
            SoftmaxCentered(),
            tfb.SoftmaxCentered(),
            jp.array([0.2, -0.4, 0.0]),
            1,
            id="softmax_centered",
        ),
        pytest.param(
            Chain([Scale(3.0), SoftmaxCentered()]),
            tfb.Chain([tfb.Scale(3.0), tfb.SoftmaxCentered()]),
            jp.array([0.2, -0.4, 0.0]),
            1,
            id="chain_softmax_centered",
        ),
        pytest.param(
            Invert(SoftmaxCentered()),
            tfb.Invert(tfb.SoftmaxCentered()),
            simplex_values,
            1,
            id="invert_softmax_centered",
        ),
    ]


def build_sampling_cases():
    return [
        pytest.param(
            Normal(0.5, 2.0),
            tfd.Normal(0.5, 2.0),
            20_000,
            jax.random.PRNGKey(3),
            id="normal_scalar",
        ),
        pytest.param(
            Normal(jp.array([0.0, 1.0]), jp.array([1.0, 2.0])),
            tfd.Normal(jp.array([0.0, 1.0]), jp.array([1.0, 2.0])),
            20_000,
            jax.random.PRNGKey(5),
            id="normal_broadcast",
        ),
        pytest.param(
            Uniform(-1.0, 2.0),
            tfd.Uniform(-1.0, 2.0),
            20_000,
            jax.random.PRNGKey(7),
            id="uniform_scalar",
        ),
        pytest.param(
            Independent(
                Normal(jp.zeros((2, 3)), jp.ones((2, 3))),
                2,
            ),
            tfd.Independent(
                tfd.Normal(jp.zeros((2, 3)), jp.ones((2, 3))),
                2,
            ),
            10_000,
            jax.random.PRNGKey(11),
            id="independent_2d",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(0.0, 1.0),
                Chain([Shift(1.5), Scale(0.5)]),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(0.0, 1.0),
                tfb.Chain([tfb.Shift(1.5), tfb.Scale(0.5)]),
            ),
            10_000,
            jax.random.PRNGKey(13),
            id="transformed_affine",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(0.0, 1.0),
                Sigmoid(-2.0, 3.0),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(0.0, 1.0),
                tfb.Sigmoid(-2.0, 3.0),
            ),
            10_000,
            jax.random.PRNGKey(17),
            id="transformed_bounded",
        ),
        pytest.param(
            Laplace(0.0, 1.5),
            tfd.Laplace(0.0, 1.5),
            20_000,
            jax.random.PRNGKey(19),
            id="laplace",
        ),
        pytest.param(
            Bernoulli(probs=0.3, dtype=jp.float32),
            tfd.Bernoulli(probs=0.3, dtype=jp.float32),
            20_000,
            jax.random.PRNGKey(23),
            id="bernoulli",
        ),
        pytest.param(
            Categorical(probs=jp.array([0.2, 0.3, 0.5])),
            tfd.Categorical(probs=jp.array([0.2, 0.3, 0.5])),
            20_000,
            jax.random.PRNGKey(29),
            id="categorical",
        ),
        pytest.param(
            MultivariateNormalDiag(
                loc=jp.array([0.0, 1.0]),
                scale_diag=jp.array([1.0, 0.5]),
            ),
            tfd.MultivariateNormalDiag(
                loc=jp.array([0.0, 1.0]),
                scale_diag=jp.array([1.0, 0.5]),
            ),
            20_000,
            jax.random.PRNGKey(31),
            id="mvn_diag",
        ),
    ]


@pytest.mark.parametrize("ours, theirs, values", build_distribution_cases())
def test_distribution_parity_with_tfp(ours, theirs, values):
    assert_distribution_parity(ours, theirs, values)


@pytest.mark.parametrize("ours, theirs, values, event_ndims", build_bijector_cases())
def test_bijector_parity_with_tfp(ours, theirs, values, event_ndims):
    assert_bijector_parity(ours, theirs, values, event_ndims)


@pytest.mark.parametrize("ours, theirs, num_samples, key", build_sampling_cases())
def test_sampling_parity_with_tfp(ours, theirs, num_samples, key):
    assert_sample_parity(ours, theirs, num_samples, key)


def test_normal_quantile_and_cdf_match_tfp():
    ours = Normal(jp.array([-0.5, 1.0]), jp.array([0.8, 1.7]))
    theirs = tfd.Normal(jp.array([-0.5, 1.0]), jp.array([0.8, 1.7]))
    values = jp.array([0.1, 0.9])
    x = jp.array([-1.0, 0.5])
    assert_method_parity(ours, theirs, "quantile", values)
    assert_method_parity(ours, theirs, "cdf", x)
    assert_method_parity(ours, theirs, "log_cdf", x)


def test_uniform_cdf_matches_tfp():
    ours = Uniform(-1.0, 2.0)
    theirs = tfd.Uniform(-1.0, 2.0)
    values = jp.array([-2.0, -1.0, 0.0, 2.0, 3.0])
    assert_method_parity(ours, theirs, "cdf", values)


@pytest.mark.parametrize(
    "ours, theirs, values, atol",
    [
        pytest.param(
            Deterministic(jp.array([0.25, -0.5])),
            tfd.Deterministic(jp.array([0.25, -0.5])),
            jp.array([0.25, 1.0]),
            1e-6,
            id="deterministic",
        ),
        pytest.param(
            Laplace(jp.array([-1.0, 0.5]), jp.array([0.4, 1.1])),
            tfd.Laplace(jp.array([-1.0, 0.5]), jp.array([0.4, 1.1])),
            jp.array([-0.8, 0.2]),
            1e-6,
            id="laplace",
        ),
        pytest.param(
            StudentT(5.0, -0.2, 1.3),
            tfd.StudentT(5.0, -0.2, 1.3),
            jp.array([-1.0, 0.5, 1.3]),
            1e-6,
            id="student_t",
        ),
        pytest.param(
            LogNormal(-0.1, 0.7),
            tfd.LogNormal(-0.1, 0.7),
            jp.array([0.2, 1.1, 2.4]),
            1e-6,
            id="log_normal",
        ),
        pytest.param(
            TruncatedNormal(-0.2, 0.8, -1.0, 1.2),
            tfd.TruncatedNormal(-0.2, 0.8, -1.0, 1.2),
            jp.array([-1.5, -0.5, 0.4, 1.2]),
            1e-6,
            id="truncated_normal",
        ),
        pytest.param(
            Beta(2.0, 3.0),
            tfd.Beta(2.0, 3.0),
            jp.array([0.2, 0.5, 0.8]),
            1e-6,
            id="beta",
        ),
        pytest.param(
            VonMises(0.3, 2.0),
            tfd.VonMises(0.3, 2.0),
            jp.array([-2.0, 0.1, 1.4]),
            1e-5,
            id="von_mises",
        ),
    ],
)
def test_scalar_distribution_value_parity(ours, theirs, values, atol):
    assert_distribution_parity(ours, theirs, values, atol=atol)


def test_truncated_normal_cdf_matches_tfp():
    ours = TruncatedNormal(0.0, 1.0, -1.0, 1.5)
    theirs = tfd.TruncatedNormal(0.0, 1.0, -1.0, 1.5)
    values = jp.array([-1.2, -1.0, 0.2, 1.5, 2.0])
    assert_method_parity(ours, theirs, "cdf", values, atol=1e-6)
    assert_method_parity(ours, theirs, "log_cdf", values, atol=1e-6)


def test_bernoulli_value_and_parameter_parity():
    probs_ours = Bernoulli(probs=jp.array([0.2, 0.7]), dtype=jp.float32)
    probs_theirs = tfd.Bernoulli(probs=jp.array([0.2, 0.7]), dtype=jp.float32)
    logits_ours = Bernoulli(logits=jp.array([-0.4, 0.9]))
    logits_theirs = tfd.Bernoulli(logits=jp.array([-0.4, 0.9]))
    values = jp.array([0.0, 1.0])
    assert_distribution_parity(probs_ours, probs_theirs, values)
    assert_distribution_parity(logits_ours, logits_theirs, values)
    assert_method_parity(probs_ours, probs_theirs, "probs_parameter")
    assert_method_parity(probs_ours, probs_theirs, "logits_parameter")
    assert_method_parity(logits_ours, logits_theirs, "probs_parameter")
    assert_method_parity(logits_ours, logits_theirs, "logits_parameter")


def test_categorical_value_and_parameter_parity():
    probs_ours = Categorical(probs=jp.array([0.2, 0.3, 0.5]))
    probs_theirs = tfd.Categorical(probs=jp.array([0.2, 0.3, 0.5]))
    logits_ours = Categorical(logits=jp.array([0.1, -0.3, 0.8]))
    logits_theirs = tfd.Categorical(logits=jp.array([0.1, -0.3, 0.8]))
    values = jp.array([0, 1, 2])
    assert_distribution_parity(probs_ours, probs_theirs, values)
    assert_distribution_parity(logits_ours, logits_theirs, values)
    assert_method_parity(probs_ours, probs_theirs, "probs_parameter")
    assert_method_parity(probs_ours, probs_theirs, "logits_parameter")
    assert_method_parity(logits_ours, logits_theirs, "probs_parameter")
    assert_method_parity(logits_ours, logits_theirs, "logits_parameter")


def test_poisson_value_parity():
    rate_ours = Poisson(rate=jp.array([2.0, 4.0]))
    rate_theirs = tfd.Poisson(rate=jp.array([2.0, 4.0]))
    log_ours = Poisson(log_rate=jp.log(jp.array([2.0, 4.0])))
    log_theirs = tfd.Poisson(log_rate=jp.log(jp.array([2.0, 4.0])))
    values = jp.array([1.0, 3.0])
    assert_distribution_parity(rate_ours, rate_theirs, values)
    assert_distribution_parity(log_ours, log_theirs, values)


def test_relaxed_one_hot_value_and_parameter_parity():
    probs = jp.array([0.2, 0.5, 0.3])
    logits = jp.log(probs)
    values = jp.array([0.15, 0.55, 0.30])
    probs_ours = RelaxedOneHotCategorical(0.7, probs=probs)
    probs_theirs = tfd.RelaxedOneHotCategorical(0.7, probs=probs)
    logits_ours = RelaxedOneHotCategorical(0.7, logits=logits)
    logits_theirs = tfd.RelaxedOneHotCategorical(0.7, logits=logits)
    assert_distribution_parity(probs_ours, probs_theirs, values, atol=1e-5)
    assert_distribution_parity(logits_ours, logits_theirs, values, atol=1e-5)
    assert_method_parity(probs_ours, probs_theirs, "probs_parameter")
    assert_method_parity(probs_ours, probs_theirs, "logits_parameter")
    assert_method_parity(logits_ours, logits_theirs, "probs_parameter")
    assert_method_parity(logits_ours, logits_theirs, "logits_parameter")


def test_multivariate_normal_diag_value_parity():
    ours = MultivariateNormalDiag(
        loc=jp.array([[0.0, 1.0], [1.5, -0.5]]),
        scale_diag=jp.array([[1.0, 0.5], [0.3, 0.8]]),
    )
    theirs = tfd.MultivariateNormalDiag(
        loc=jp.array([[0.0, 1.0], [1.5, -0.5]]),
        scale_diag=jp.array([[1.0, 0.5], [0.3, 0.8]]),
    )
    values = jp.array([[0.2, 1.4], [1.1, -0.1]])
    assert_distribution_parity(ours, theirs, values, atol=1e-6)
    assert_method_parity(ours, theirs, "variance")
    assert_method_parity(ours, theirs, "covariance")


def test_multivariate_normal_full_value_parity():
    covariance = jp.array(
        [[[0.6, 0.1], [0.1, 0.5]], [[0.4, -0.05], [-0.05, 0.7]]]
    )
    ours = MultivariateNormalFullCovariance(
        loc=jp.array([[0.0, 1.0], [1.5, -0.5]]),
        covariance_matrix=covariance,
    )
    theirs = tfd.MultivariateNormalFullCovariance(
        loc=jp.array([[0.0, 1.0], [1.5, -0.5]]),
        covariance_matrix=covariance,
    )
    values = jp.array([[0.2, 1.4], [1.1, -0.1]])
    assert_distribution_parity(ours, theirs, values, atol=1e-5)
    assert_method_parity(ours, theirs, "covariance", atol=1e-6)


def test_mixture_same_family_value_parity():
    weights = jp.array([0.3, 0.7])
    means = jp.array([-1.0, 1.5])
    scales = jp.array([0.4, 0.9])
    ours = MixtureSameFamily(
        Categorical(probs=weights),
        Normal(loc=means, scale=scales),
    )
    theirs = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights),
        tfd.Normal(loc=means, scale=scales),
    )
    values = jp.array([-1.2, 0.0, 1.7])
    assert_distribution_parity(ours, theirs, values, atol=1e-6)


def test_quantized_distribution_value_parity():
    shift = 1.5
    scale = 2.0
    ours = QuantizedDistribution(
        TransformedDistribution(
            Normal(0.0, 1.0),
            Chain([Shift(-0.5), Shift(shift), Scale(scale)]),
        )
    )
    theirs = tfd.QuantizedDistribution(
        tfd.TransformedDistribution(
            tfd.Normal(0.0, 1.0),
            tfb.Chain(
                [
                    tfb.Shift(-0.5),
                    tfb.Shift(shift),
                    tfb.Scale(scale),
                ]
            ),
        )
    )
    values = jp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    assert_distribution_parity(ours, theirs, values, atol=1e-6)
    assert_method_parity(ours, theirs, "cdf", values, atol=1e-6)
    assert_method_parity(ours, theirs, "log_cdf", values, atol=1e-6)
