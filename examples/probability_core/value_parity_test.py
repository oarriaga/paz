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
from .parity_helpers import (
    assert_bijector_parity,
    assert_distribution_parity,
    assert_method_parity,
    assert_sample_parity,
)

tfd = tfp.distributions
tfb = tfp.bijectors


def as_float(values):
    return jp.asarray(values, dtype=jp.float64)


def build_distribution_cases():
    bounded_values = tfb.Sigmoid(as_float(-2.0), as_float(3.0))(
        as_float([-1.5, 0.0, 1.2])
    )
    return [
        pytest.param(
            Normal(as_float(0.5), as_float(2.0)),
            tfd.Normal(as_float(0.5), as_float(2.0)),
            as_float([-1.5, -0.2, 0.3, 2.0]),
            id="normal_scalar",
        ),
        pytest.param(
            Normal(as_float([0.0, 1.0]), as_float([1.0, 2.0])),
            tfd.Normal(as_float([0.0, 1.0]), as_float([1.0, 2.0])),
            as_float([[0.2, 1.3], [-0.5, 2.7]]),
            id="normal_broadcast",
        ),
        pytest.param(
            Uniform(as_float(0.1), as_float(0.9)),
            tfd.Uniform(as_float(0.1), as_float(0.9)),
            as_float([-0.2, 0.1, 0.4, 0.9, 1.2]),
            id="uniform_scalar",
        ),
        pytest.param(
            Independent(
                Normal(as_float([0.0, 1.0]), as_float([1.0, 2.0])),
                1,
            ),
            tfd.Independent(
                tfd.Normal(as_float([0.0, 1.0]), as_float([1.0, 2.0])),
                1,
            ),
            as_float([0.2, 1.3]),
            id="independent_1d",
        ),
        pytest.param(
            Independent(Normal(as_float(jp.zeros((2, 3))),
                               as_float(jp.ones((2, 3)))), 2),
            tfd.Independent(
                tfd.Normal(as_float(jp.zeros((2, 3))),
                           as_float(jp.ones((2, 3)))), 2
            ),
            as_float([[0.2, -0.3, 0.8], [1.1, -1.0, 0.4]]),
            id="independent_2d",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(as_float(0.0), as_float(1.0)),
                Chain([Shift(as_float(1.5)), Scale(as_float(0.5))]),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(as_float(0.0), as_float(1.0)),
                tfb.Chain([
                    tfb.Shift(as_float(1.5)),
                    tfb.Scale(as_float(0.5)),
                ]),
            ),
            as_float([0.5, 1.0, 1.5]),
            id="transformed_affine",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(as_float(0.0), as_float(1.0)),
                Sigmoid(as_float(-2.0), as_float(3.0)),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(as_float(0.0), as_float(1.0)),
                tfb.Sigmoid(as_float(-2.0), as_float(3.0)),
            ),
            bounded_values,
            id="transformed_bounded",
        ),
    ]


def build_bijector_cases():
    simplex_values = jax.nn.softmax(as_float([0.2, -0.4, 0.0, -0.3]))
    return [
        pytest.param(
            Identity(), tfb.Identity(), as_float([-2.0, -0.1, 1.7]), 0,
            id="identity",
        ),
        pytest.param(
            Shift(as_float(1.2)), tfb.Shift(as_float(1.2)),
            as_float([-2.0, -0.1, 1.7]), 0, id="shift",
        ),
        pytest.param(
            Scale(as_float(-0.7)), tfb.Scale(as_float(-0.7)),
            as_float([-2.0, -0.1, 1.7]), 0, id="scale",
        ),
        pytest.param(
            Sigmoid(), tfb.Sigmoid(), as_float([-2.0, -0.1, 1.7]), 0,
            id="sigmoid",
        ),
        pytest.param(
            Sigmoid(as_float(-2.0), as_float(3.0)),
            tfb.Sigmoid(as_float(-2.0), as_float(3.0)),
            as_float([-2.0, -0.1, 1.7]),
            0,
            id="sigmoid_bounded",
        ),
        pytest.param(
            Chain([Shift(as_float(1.0)), Scale(as_float(0.5)), Sigmoid()]),
            tfb.Chain([
                tfb.Shift(as_float(1.0)),
                tfb.Scale(as_float(0.5)),
                tfb.Sigmoid(),
            ]),
            as_float([-2.0, -0.1, 1.7]),
            0,
            id="chain_affine_sigmoid",
        ),
        pytest.param(
            Chain([Shift(as_float(-1.0)), Scale(as_float(2.5))]),
            tfb.Chain([
                tfb.Shift(as_float(-1.0)),
                tfb.Scale(as_float(2.5)),
            ]),
            as_float([-1.5, 0.0, 2.0]),
            0,
            id="chain_affine",
        ),
        pytest.param(
            Exp(), tfb.Exp(), as_float([-1.0, 0.2, 1.5]), 0, id="exp",
        ),
        pytest.param(
            Softplus(), tfb.Softplus(), as_float([-1.0, 0.2, 1.5]), 0,
            id="softplus",
        ),
        pytest.param(
            Softplus(hinge_softness=as_float(0.3), low=as_float(0.2)),
            tfb.Softplus(
                hinge_softness=as_float(0.3), low=as_float(0.2)
            ),
            as_float([-1.0, 0.2, 1.5]),
            0,
            id="softplus_shifted",
        ),
        pytest.param(
            Invert(Scale(as_float(3.0))),
            tfb.Invert(tfb.Scale(as_float(3.0))),
            as_float([-1.5, 0.0, 2.0]),
            0,
            id="invert_scale",
        ),
        pytest.param(
            SoftmaxCentered(),
            tfb.SoftmaxCentered(),
            as_float([0.2, -0.4, 0.0]),
            1,
            id="softmax_centered",
        ),
        pytest.param(
            Chain([Scale(as_float(3.0)), SoftmaxCentered()]),
            tfb.Chain([tfb.Scale(as_float(3.0)), tfb.SoftmaxCentered()]),
            as_float([0.2, -0.4, 0.0]),
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
            Normal(as_float(0.5), as_float(2.0)),
            tfd.Normal(as_float(0.5), as_float(2.0)),
            20_000,
            jax.random.PRNGKey(3),
            id="normal_scalar",
        ),
        pytest.param(
            Normal(as_float([0.0, 1.0]), as_float([1.0, 2.0])),
            tfd.Normal(as_float([0.0, 1.0]), as_float([1.0, 2.0])),
            20_000,
            jax.random.PRNGKey(5),
            id="normal_broadcast",
        ),
        pytest.param(
            Uniform(as_float(-1.0), as_float(2.0)),
            tfd.Uniform(as_float(-1.0), as_float(2.0)),
            20_000,
            jax.random.PRNGKey(7),
            id="uniform_scalar",
        ),
        pytest.param(
            Independent(
                Normal(as_float(jp.zeros((2, 3))), as_float(jp.ones((2, 3)))),
                2,
            ),
            tfd.Independent(
                tfd.Normal(as_float(jp.zeros((2, 3))),
                           as_float(jp.ones((2, 3)))), 2
            ),
            10_000,
            jax.random.PRNGKey(11),
            id="independent_2d",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(as_float(0.0), as_float(1.0)),
                Chain([Shift(as_float(1.5)), Scale(as_float(0.5))]),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(as_float(0.0), as_float(1.0)),
                tfb.Chain([
                    tfb.Shift(as_float(1.5)),
                    tfb.Scale(as_float(0.5)),
                ]),
            ),
            10_000,
            jax.random.PRNGKey(13),
            id="transformed_affine",
        ),
        pytest.param(
            TransformedDistribution(
                Normal(as_float(0.0), as_float(1.0)),
                Sigmoid(as_float(-2.0), as_float(3.0)),
            ),
            tfd.TransformedDistribution(
                tfd.Normal(as_float(0.0), as_float(1.0)),
                tfb.Sigmoid(as_float(-2.0), as_float(3.0)),
            ),
            10_000,
            jax.random.PRNGKey(17),
            id="transformed_bounded",
        ),
        pytest.param(
            Laplace(as_float(0.0), as_float(1.5)),
            tfd.Laplace(as_float(0.0), as_float(1.5)),
            20_000,
            jax.random.PRNGKey(19),
            id="laplace",
        ),
        pytest.param(
            Bernoulli(probs=as_float(0.3), dtype=jp.float64),
            tfd.Bernoulli(probs=as_float(0.3), dtype=jp.float64),
            20_000,
            jax.random.PRNGKey(23),
            id="bernoulli",
        ),
        pytest.param(
            Categorical(probs=as_float([0.2, 0.3, 0.5])),
            tfd.Categorical(probs=as_float([0.2, 0.3, 0.5])),
            20_000,
            jax.random.PRNGKey(29),
            id="categorical",
        ),
        pytest.param(
            MultivariateNormalDiag(
                loc=as_float([0.0, 1.0]), scale_diag=as_float([1.0, 0.5])
            ),
            tfd.MultivariateNormalDiag(
                loc=as_float([0.0, 1.0]), scale_diag=as_float([1.0, 0.5])
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
    ours = Normal(as_float([-0.5, 1.0]), as_float([0.8, 1.7]))
    theirs = tfd.Normal(as_float([-0.5, 1.0]), as_float([0.8, 1.7]))
    assert_method_parity(ours, theirs, "quantile", as_float([0.1, 0.9]))
    assert_method_parity(ours, theirs, "cdf", as_float([-1.0, 0.5]))
    assert_method_parity(ours, theirs, "log_cdf", as_float([-1.0, 0.5]))


def test_uniform_cdf_matches_tfp():
    ours = Uniform(as_float(-1.0), as_float(2.0))
    theirs = tfd.Uniform(as_float(-1.0), as_float(2.0))
    values = as_float([-2.0, -1.0, 0.0, 2.0, 3.0])
    assert_method_parity(ours, theirs, "cdf", values)


@pytest.mark.parametrize(
    "ours, theirs, values, atol",
    [
        pytest.param(
            Deterministic(as_float([0.25, -0.5])),
            tfd.Deterministic(as_float([0.25, -0.5])),
            as_float([0.25, 1.0]),
            1e-6,
            id="deterministic",
        ),
        pytest.param(
            Laplace(as_float([-1.0, 0.5]), as_float([0.4, 1.1])),
            tfd.Laplace(as_float([-1.0, 0.5]), as_float([0.4, 1.1])),
            as_float([-0.8, 0.2]),
            1e-6,
            id="laplace",
        ),
        pytest.param(
            StudentT(as_float(5.0), as_float(-0.2), as_float(1.3)),
            tfd.StudentT(as_float(5.0), as_float(-0.2), as_float(1.3)),
            as_float([-1.0, 0.5, 1.3]),
            1e-6,
            id="student_t",
        ),
        pytest.param(
            LogNormal(as_float(-0.1), as_float(0.7)),
            tfd.LogNormal(as_float(-0.1), as_float(0.7)),
            as_float([0.2, 1.1, 2.4]),
            1e-6,
            id="log_normal",
        ),
        pytest.param(
            TruncatedNormal(
                as_float(-0.2),
                as_float(0.8),
                as_float(-1.0),
                as_float(1.2),
            ),
            tfd.TruncatedNormal(
                as_float(-0.2),
                as_float(0.8),
                as_float(-1.0),
                as_float(1.2),
            ),
            as_float([-1.5, -0.5, 0.4, 1.2]),
            1e-6,
            id="truncated_normal",
        ),
        pytest.param(
            Beta(as_float(2.0), as_float(3.0)),
            tfd.Beta(as_float(2.0), as_float(3.0)),
            as_float([0.2, 0.5, 0.8]),
            1e-6,
            id="beta",
        ),
        pytest.param(
            VonMises(as_float(0.3), as_float(2.0)),
            tfd.VonMises(as_float(0.3), as_float(2.0)),
            as_float([-2.0, 0.1, 1.4]),
            1e-5,
            id="von_mises",
        ),
    ],
)
def test_scalar_distribution_value_parity(ours, theirs, values, atol):
    assert_distribution_parity(ours, theirs, values, atol=atol)


def test_truncated_normal_cdf_matches_tfp():
    ours = TruncatedNormal(
        as_float(0.0), as_float(1.0), as_float(-1.0), as_float(1.5)
    )
    theirs = tfd.TruncatedNormal(
        as_float(0.0), as_float(1.0), as_float(-1.0), as_float(1.5)
    )
    values = as_float([-1.2, -1.0, 0.2, 1.5, 2.0])
    assert_method_parity(ours, theirs, "cdf", values)
    assert_method_parity(ours, theirs, "log_cdf", values)


def test_bernoulli_value_and_parameter_parity():
    probs = as_float([0.2, 0.7])
    logits = as_float([-0.4, 0.9])
    probs_ours = Bernoulli(probs=probs, dtype=jp.float64)
    probs_theirs = tfd.Bernoulli(probs=probs, dtype=jp.float64)
    logits_ours = Bernoulli(logits=logits)
    logits_theirs = tfd.Bernoulli(logits=logits)
    values = as_float([0.0, 1.0])
    assert_distribution_parity(probs_ours, probs_theirs, values)
    assert_distribution_parity(logits_ours, logits_theirs, values)
    assert_method_parity(probs_ours, probs_theirs, "probs_parameter")
    assert_method_parity(probs_ours, probs_theirs, "logits_parameter")
    assert_method_parity(logits_ours, logits_theirs, "probs_parameter")
    assert_method_parity(logits_ours, logits_theirs, "logits_parameter")


def test_categorical_value_and_parameter_parity():
    probs_ours = Categorical(probs=as_float([0.2, 0.3, 0.5]))
    probs_theirs = tfd.Categorical(probs=as_float([0.2, 0.3, 0.5]))
    logits_ours = Categorical(logits=as_float([0.1, -0.3, 0.8]))
    logits_theirs = tfd.Categorical(logits=as_float([0.1, -0.3, 0.8]))
    values = jp.array([0, 1, 2])
    assert_distribution_parity(probs_ours, probs_theirs, values)
    assert_distribution_parity(logits_ours, logits_theirs, values)
    assert_method_parity(probs_ours, probs_theirs, "probs_parameter")
    assert_method_parity(probs_ours, probs_theirs, "logits_parameter")
    assert_method_parity(logits_ours, logits_theirs, "probs_parameter")
    assert_method_parity(logits_ours, logits_theirs, "logits_parameter")


def test_poisson_value_parity():
    rates = as_float([2.0, 4.0])
    rate_ours = Poisson(rate=rates)
    rate_theirs = tfd.Poisson(rate=rates)
    log_ours = Poisson(log_rate=jp.log(rates))
    log_theirs = tfd.Poisson(log_rate=jp.log(rates))
    values = as_float([1.0, 3.0])
    assert_distribution_parity(rate_ours, rate_theirs, values)
    assert_distribution_parity(log_ours, log_theirs, values)


def test_relaxed_one_hot_value_and_parameter_parity():
    probs = as_float([0.2, 0.5, 0.3])
    logits = jp.log(probs)
    values = as_float([0.15, 0.55, 0.30])
    probs_ours = RelaxedOneHotCategorical(as_float(0.7), probs=probs)
    probs_theirs = tfd.RelaxedOneHotCategorical(as_float(0.7), probs=probs)
    logits_ours = RelaxedOneHotCategorical(as_float(0.7), logits=logits)
    logits_theirs = tfd.RelaxedOneHotCategorical(as_float(0.7), logits=logits)
    assert_distribution_parity(probs_ours, probs_theirs, values, atol=1e-5)
    assert_distribution_parity(logits_ours, logits_theirs, values, atol=1e-5)
    assert_method_parity(probs_ours, probs_theirs, "probs_parameter")
    assert_method_parity(probs_ours, probs_theirs, "logits_parameter")
    assert_method_parity(logits_ours, logits_theirs, "probs_parameter")
    assert_method_parity(logits_ours, logits_theirs, "logits_parameter")


def test_multivariate_normal_diag_value_parity():
    ours = MultivariateNormalDiag(
        loc=as_float([[0.0, 1.0], [1.5, -0.5]]),
        scale_diag=as_float([[1.0, 0.5], [0.3, 0.8]]),
    )
    theirs = tfd.MultivariateNormalDiag(
        loc=as_float([[0.0, 1.0], [1.5, -0.5]]),
        scale_diag=as_float([[1.0, 0.5], [0.3, 0.8]]),
    )
    values = as_float([[0.2, 1.4], [1.1, -0.1]])
    assert_distribution_parity(ours, theirs, values)
    assert_method_parity(ours, theirs, "variance")
    assert_method_parity(ours, theirs, "covariance")


def test_multivariate_normal_full_value_parity():
    covariance = as_float(
        [[[0.6, 0.1], [0.1, 0.5]], [[0.4, -0.05], [-0.05, 0.7]]]
    )
    ours = MultivariateNormalFullCovariance(
        loc=as_float([[0.0, 1.0], [1.5, -0.5]]),
        covariance_matrix=covariance,
    )
    theirs = tfd.MultivariateNormalFullCovariance(
        loc=as_float([[0.0, 1.0], [1.5, -0.5]]),
        covariance_matrix=covariance,
    )
    values = as_float([[0.2, 1.4], [1.1, -0.1]])
    assert_distribution_parity(ours, theirs, values, atol=1e-5)
    assert_method_parity(ours, theirs, "covariance")


def test_mixture_same_family_value_parity():
    weights = as_float([0.3, 0.7])
    means = as_float([-1.0, 1.5])
    scales = as_float([0.4, 0.9])
    ours = MixtureSameFamily(
        Categorical(probs=weights), Normal(loc=means, scale=scales)
    )
    theirs = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights), tfd.Normal(loc=means, scale=scales)
    )
    values = as_float([-1.2, 0.0, 1.7])
    assert_distribution_parity(ours, theirs, values)


def test_quantized_distribution_value_parity():
    ours = QuantizedDistribution(
        TransformedDistribution(
            Normal(as_float(0.0), as_float(1.0)),
            Chain([
                Shift(as_float(-0.5)),
                Shift(as_float(1.5)),
                Scale(as_float(2.0)),
            ]),
        )
    )
    theirs = tfd.QuantizedDistribution(
        tfd.TransformedDistribution(
            tfd.Normal(as_float(0.0), as_float(1.0)),
            tfb.Chain([
                tfb.Shift(as_float(-0.5)),
                tfb.Shift(as_float(1.5)),
                tfb.Scale(as_float(2.0)),
            ]),
        )
    )
    values = as_float([-1.0, 0.0, 1.0, 2.0, 3.0])
    assert_distribution_parity(ours, theirs, values)
    assert_method_parity(ours, theirs, "cdf", values)
    assert_method_parity(ours, theirs, "log_cdf", values)
