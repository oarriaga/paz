import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from .bijectors import Chain, Exp, Invert, Scale, Shift, Sigmoid
from .bijectors import SoftmaxCentered, Softplus
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
    assert_gradient_parity,
    assert_method_gradient_parity,
    assert_sample_gradient_parity,
)

tfd = tfp.distributions
tfb = tfp.bijectors


def as_float(values):
    return jp.asarray(values, dtype=jp.float64)


def test_normal_gradient_parity():
    build_ours = lambda loc, scale: Normal(loc, scale)
    build_theirs = lambda loc, scale: tfd.Normal(loc, scale)
    values = (as_float([-0.4, 0.3, 1.2]),)
    params = (0.2, 1.3)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "prob", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "cdf", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_cdf", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "survival_function", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_survival_function", params, values
    )
    assert_method_gradient_parity(
        build_ours,
        build_theirs,
        "quantile",
        params,
        (as_float([0.2, 0.8]),),
    )


def test_uniform_gradient_parity():
    build_ours = lambda low, high: Uniform(low, high)
    build_theirs = lambda low, high: tfd.Uniform(low, high)
    params = (-0.5, 1.7)
    values = (as_float([0.0, 0.3, 1.1]),)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "cdf", params, values
    )


def test_laplace_gradient_parity():
    build_ours = lambda loc, scale: Laplace(loc, scale)
    build_theirs = lambda loc, scale: tfd.Laplace(loc, scale)
    params = (0.1, 1.1)
    values = (as_float([-1.0, 0.4, 1.8]),)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "cdf", params, values
    )


def test_student_t_gradient_parity():
    build_ours = lambda df, loc, scale: StudentT(df, loc, scale)
    build_theirs = lambda df, loc, scale: tfd.StudentT(df, loc, scale)
    params = (5.0, 0.1, 0.9)
    values = (as_float([-0.8, 0.3, 1.1]),)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values
    )


def test_log_normal_gradient_parity():
    build_ours = lambda loc, scale: LogNormal(loc, scale)
    build_theirs = lambda loc, scale: tfd.LogNormal(loc, scale)
    params = (0.1, 0.7)
    values = (as_float([0.2, 1.1, 2.7]),)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "cdf", params, values
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_cdf", params, values
    )


def test_truncated_normal_gradient_parity():
    build_ours = lambda loc, scale, low, high: TruncatedNormal(
        loc, scale, low, high
    )
    build_theirs = lambda loc, scale, low, high: tfd.TruncatedNormal(
        loc, scale, low, high
    )
    params = (0.0, 0.8, -1.0, 1.3)
    values = (as_float([-0.7, 0.1, 0.9]),)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values, atol=1e-8
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "cdf", params, values, atol=1e-8
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_cdf", params, values, atol=1e-8
    )


def test_beta_gradient_parity():
    build_ours = lambda alpha, beta: Beta(alpha, beta)
    build_theirs = lambda alpha, beta: tfd.Beta(alpha, beta)
    params = (2.0, 3.0)
    values = (as_float([0.2, 0.5, 0.8]),)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values, atol=1e-8
    )


def test_von_mises_gradient_parity():
    build_ours = lambda loc, concentration: VonMises(loc, concentration)
    build_theirs = lambda loc, concentration: tfd.VonMises(
        loc, concentration
    )
    params = (0.2, 2.0)
    values = (as_float([-2.1, 0.4, 1.7]),)
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values, atol=1e-8
    )


def test_bernoulli_gradient_parity():
    values = (as_float([0.0, 1.0]),)
    build_probs_ours = lambda probs: Bernoulli(probs=probs, dtype=jp.float64)
    build_probs_theirs = lambda probs: tfd.Bernoulli(
        probs=probs, dtype=jp.float64
    )
    build_logits_ours = lambda logits: Bernoulli(logits=logits)
    build_logits_theirs = lambda logits: tfd.Bernoulli(logits=logits)
    params = (as_float([0.2, 0.7]),)
    assert_method_gradient_parity(
        build_probs_ours, build_probs_theirs, "log_prob", params, values
    )
    assert_method_gradient_parity(
        build_probs_ours, build_probs_theirs, "probs_parameter", params
    )
    assert_method_gradient_parity(
        build_probs_ours, build_probs_theirs, "logits_parameter", params
    )
    logits = (as_float([-0.4, 0.9]),)
    assert_method_gradient_parity(
        build_logits_ours, build_logits_theirs, "log_prob", logits, values
    )
    assert_method_gradient_parity(
        build_logits_ours, build_logits_theirs, "probs_parameter", logits
    )
    assert_method_gradient_parity(
        build_logits_ours, build_logits_theirs, "logits_parameter", logits
    )


def test_categorical_gradient_parity():
    values = (jp.array([0, 2]),)
    build_probs_ours = lambda probs: Categorical(probs=probs)
    build_probs_theirs = lambda probs: tfd.Categorical(probs=probs)
    build_logits_ours = lambda logits: Categorical(logits=logits)
    build_logits_theirs = lambda logits: tfd.Categorical(logits=logits)
    params = (as_float([0.2, 0.3, 0.5]),)
    assert_method_gradient_parity(
        build_probs_ours, build_probs_theirs, "log_prob", params, values
    )
    assert_method_gradient_parity(
        build_probs_ours, build_probs_theirs, "probs_parameter", params
    )
    assert_method_gradient_parity(
        build_probs_ours, build_probs_theirs, "logits_parameter", params
    )
    logits = (as_float([0.1, -0.3, 0.8]),)
    assert_method_gradient_parity(
        build_logits_ours, build_logits_theirs, "log_prob", logits, values
    )
    assert_method_gradient_parity(
        build_logits_ours, build_logits_theirs, "probs_parameter", logits
    )
    assert_method_gradient_parity(
        build_logits_ours, build_logits_theirs, "logits_parameter", logits
    )


def test_poisson_gradient_parity():
    values = (as_float([1.0, 3.0]),)
    build_rate_ours = lambda rate: Poisson(rate=rate)
    build_rate_theirs = lambda rate: tfd.Poisson(rate=rate)
    build_log_ours = lambda log_rate: Poisson(log_rate=log_rate)
    build_log_theirs = lambda log_rate: tfd.Poisson(log_rate=log_rate)
    assert_method_gradient_parity(
        build_rate_ours,
        build_rate_theirs,
        "log_prob",
        (as_float([2.0, 4.0]),),
        values,
    )
    assert_method_gradient_parity(
        build_log_ours,
        build_log_theirs,
        "log_prob",
        (jp.log(as_float([2.0, 4.0])),),
        values,
    )


def test_relaxed_one_hot_gradient_parity():
    values = (as_float([0.2, 0.5, 0.3]),)
    build_ours = lambda temperature, logits: RelaxedOneHotCategorical(
        temperature, logits=logits
    )
    build_theirs = lambda temperature, logits: tfd.RelaxedOneHotCategorical(
        temperature, logits=logits
    )
    params = (as_float(0.7), as_float([0.1, 0.4, -0.2]))
    assert_method_gradient_parity(
        build_ours, build_theirs, "log_prob", params, values, atol=1e-8
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "probs_parameter", params
    )
    assert_method_gradient_parity(
        build_ours, build_theirs, "logits_parameter", params
    )


def test_multivariate_gradient_parity():
    build_diag_ours = lambda loc, scale: MultivariateNormalDiag(loc, scale)
    build_diag_theirs = lambda loc, scale: tfd.MultivariateNormalDiag(
        loc=loc, scale_diag=scale
    )
    params = (
        as_float([0.1, -0.2]),
        as_float([1.2, 0.8]),
    )
    values = (as_float([0.3, 0.4]),)
    assert_method_gradient_parity(
        build_diag_ours, build_diag_theirs, "log_prob", params, values
    )
    assert_method_gradient_parity(
        build_diag_ours, build_diag_theirs, "variance", params
    )
    assert_method_gradient_parity(
        build_diag_ours, build_diag_theirs, "covariance", params
    )

    build_full_ours = lambda loc, covariance: MultivariateNormalFullCovariance(
        loc, covariance
    )
    build_full_theirs = lambda loc, covariance: (
        tfd.MultivariateNormalFullCovariance(
            loc=loc, covariance_matrix=covariance
        )
    )
    params = (
        as_float([0.1, -0.2]),
        as_float([[1.3, 0.2], [0.2, 0.9]]),
    )
    assert_method_gradient_parity(
        build_full_ours,
        build_full_theirs,
        "log_prob",
        params,
        values,
        atol=1e-8,
    )
    assert_method_gradient_parity(
        build_full_ours, build_full_theirs, "covariance", params
    )


def test_wrapper_gradient_parity():
    build_ind_ours = lambda loc, scale: Independent(Normal(loc, scale), 1)
    build_ind_theirs = lambda loc, scale: tfd.Independent(
        tfd.Normal(loc, scale), 1
    )
    params = (as_float([0.1, 0.4]), as_float([0.9, 1.1]))
    values = (as_float([0.0, 0.6]),)
    assert_method_gradient_parity(
        build_ind_ours, build_ind_theirs, "log_prob", params, values
    )

    build_tx_ours = lambda loc, scale, shift, width: (
        TransformedDistribution(
            Normal(loc, scale),
            Chain([
                Shift(shift),
                Scale(width),
                Sigmoid(as_float(-1.0), as_float(2.0)),
            ]),
        )
    )
    build_tx_theirs = lambda loc, scale, shift, width: (
        tfd.TransformedDistribution(
            tfd.Normal(loc, scale),
            tfb.Chain(
                [
                    tfb.Shift(shift),
                    tfb.Scale(width),
                    tfb.Sigmoid(as_float(-1.0), as_float(2.0)),
                ]
            ),
        )
    )
    params = (as_float(0.1), as_float(0.7), as_float(0.3), as_float(1.2))
    values = (as_float([-0.3, 0.4, 1.2]),)
    assert_method_gradient_parity(
        build_tx_ours, build_tx_theirs, "log_prob", params, values
    )


def test_mixture_and_quantized_gradient_parity():
    build_mix_ours = lambda logits, means, scales: MixtureSameFamily(
        Categorical(logits=logits), Normal(means, scales)
    )
    build_mix_theirs = lambda logits, means, scales: tfd.MixtureSameFamily(
        tfd.Categorical(logits=logits), tfd.Normal(means, scales)
    )
    params = (
        as_float([0.2, -0.1]),
        as_float([-1.0, 1.3]),
        as_float([0.6, 0.9]),
    )
    values = (as_float([-0.2, 0.5, 1.8]),)
    assert_method_gradient_parity(
        build_mix_ours, build_mix_theirs, "log_prob", params, values
    )

    build_quantized_ours = lambda loc, scale: QuantizedDistribution(
        TransformedDistribution(
            Normal(loc, scale),
            Chain([
                Shift(as_float(-0.5)),
                Shift(as_float(1.5)),
                Scale(as_float(2.0)),
            ]),
        )
    )
    build_quantized_theirs = lambda loc, scale: tfd.QuantizedDistribution(
        tfd.TransformedDistribution(
            tfd.Normal(loc, scale),
            tfb.Chain(
                [
                    tfb.Shift(as_float(-0.5)),
                    tfb.Shift(as_float(1.5)),
                    tfb.Scale(as_float(2.0)),
                ]
            ),
        )
    )
    params = (as_float(0.1), as_float(0.8))
    values = (as_float([0.0, 1.0, 2.0]),)
    assert_method_gradient_parity(
        build_quantized_ours,
        build_quantized_theirs,
        "log_prob",
        params,
        values,
        atol=1e-8,
    )


def test_basic_bijector_gradient_parity():
    values = (as_float([-0.8, 0.2, 1.4]),)
    assert_method_gradient_parity(
        lambda shift: Shift(shift),
        lambda shift: tfb.Shift(shift),
        "__call__",
        (as_float(0.4),),
        values,
    )
    assert_method_gradient_parity(
        lambda scale: Scale(scale),
        lambda scale: tfb.Scale(scale),
        "__call__",
        (as_float(1.3),),
        values,
    )
    assert_method_gradient_parity(
        lambda scale: Scale(scale),
        lambda scale: tfb.Scale(scale),
        "forward_log_det_jacobian",
        (as_float(1.3),),
        values + (0,),
    )
    assert_method_gradient_parity(
        lambda low, high: Sigmoid(low, high),
        lambda low, high: tfb.Sigmoid(low, high),
        "inverse",
        (as_float(-1.0), as_float(2.0)),
        (as_float([-0.8, 0.1, 1.5]),),
    )
    assert_method_gradient_parity(
        lambda softness, low: Softplus(softness, low),
        lambda softness, low: tfb.Softplus(
            hinge_softness=softness, low=low
        ),
        "inverse",
        (as_float(0.3), as_float(0.2)),
        (as_float([0.3, 1.0, 4.0]),),
    )
    assert_method_gradient_parity(
        lambda softness, low: Softplus(softness, low),
        lambda softness, low: tfb.Softplus(
            hinge_softness=softness, low=low
        ),
        "forward_log_det_jacobian",
        (as_float(0.3), as_float(0.2)),
        values + (0,),
    )
    assert_method_gradient_parity(
        lambda scale: Invert(Scale(scale)),
        lambda scale: tfb.Invert(tfb.Scale(scale)),
        "inverse",
        (as_float(1.7),),
        (as_float([-0.3, 0.4, 1.2]),),
    )

    def our_loss(values):
        return jp.sum(Exp().inverse(values))

    def their_loss(values):
        return jp.sum(tfb.Exp().inverse(values))

    assert_gradient_parity(
        our_loss, their_loss, (as_float([0.3, 1.0, 4.0]),)
    )


def test_composite_bijector_gradient_parity():
    def our_forward_loss(values):
        return jp.sum(SoftmaxCentered()(values))

    def their_forward_loss(values):
        return jp.sum(tfb.SoftmaxCentered()(values))

    assert_gradient_parity(
        our_forward_loss, their_forward_loss, (as_float([0.2, -0.4, 0.0]),)
    )

    simplex = jax.nn.softmax(as_float([0.2, -0.4, 0.0, 0.1]))

    def our_inverse_loss(values):
        return jp.sum(SoftmaxCentered().inverse(values))

    def their_inverse_loss(values):
        return jp.sum(tfb.SoftmaxCentered().inverse(values))

    assert_gradient_parity(our_inverse_loss, their_inverse_loss, (simplex,))

    def our_log_det_loss(values):
        return jp.sum(SoftmaxCentered().forward_log_det_jacobian(values, 1))

    def their_log_det_loss(values):
        return jp.sum(
            tfb.SoftmaxCentered().forward_log_det_jacobian(values, 1)
        )

    assert_gradient_parity(
        our_log_det_loss, their_log_det_loss, (as_float([0.2, -0.4, 0.0]),)
    )

    assert_method_gradient_parity(
        lambda shift, scale, low, high: Chain(
            [Shift(shift), Scale(scale), Sigmoid(low, high)]
        ),
        lambda shift, scale, low, high: tfb.Chain(
            [tfb.Shift(shift), tfb.Scale(scale), tfb.Sigmoid(low, high)]
        ),
        "forward_log_det_jacobian",
        (as_float(0.2), as_float(1.3), as_float(-1.0), as_float(2.0)),
        (as_float([-0.8, 0.2, 1.1]), 0),
        atol=1e-8,
    )


def test_reparameterized_sample_gradient_parity():
    cases = [
        (
            lambda loc: Deterministic(loc),
            lambda loc: tfd.Deterministic(loc),
            (as_float(0.4),),
            jax.random.PRNGKey(1),
            1e-10,
        ),
        (
            lambda loc, scale: Normal(loc, scale),
            lambda loc, scale: tfd.Normal(loc, scale),
            (as_float(0.2), as_float(1.3)),
            jax.random.PRNGKey(3),
            1e-10,
        ),
        (
            lambda low, high: Uniform(low, high),
            lambda low, high: tfd.Uniform(low, high),
            (as_float(-0.5), as_float(1.3)),
            jax.random.PRNGKey(5),
            1e-10,
        ),
        (
            lambda loc, scale: Laplace(loc, scale),
            lambda loc, scale: tfd.Laplace(loc, scale),
            (as_float(0.2), as_float(1.1)),
            jax.random.PRNGKey(7),
            1e-10,
        ),
        (
            lambda loc, scale: LogNormal(loc, scale),
            lambda loc, scale: tfd.LogNormal(loc, scale),
            (as_float(0.1), as_float(0.7)),
            jax.random.PRNGKey(9),
            1e-10,
        ),
            (
                lambda loc, scale: TruncatedNormal(
                    loc, scale, as_float(-1.0), as_float(1.0)
                ),
                lambda loc, scale: tfd.TruncatedNormal(
                    loc, scale, as_float(-1.0), as_float(1.0)
                ),
                (as_float(0.0), as_float(0.8)),
                jax.random.PRNGKey(11),
                1e-7,
            ),
        (
            lambda temperature: RelaxedOneHotCategorical(
                temperature, probs=as_float([0.2, 0.5, 0.3])
            ),
            lambda temperature: tfd.RelaxedOneHotCategorical(
                temperature, probs=as_float([0.2, 0.5, 0.3])
            ),
            (as_float(0.7),),
            jax.random.PRNGKey(13),
            1e-10,
        ),
        (
            lambda loc, scale: MultivariateNormalDiag(loc, scale),
            lambda loc, scale: tfd.MultivariateNormalDiag(
                loc=loc, scale_diag=scale
            ),
            (as_float([0.1, -0.2]), as_float([1.2, 0.8])),
            jax.random.PRNGKey(15),
            1e-10,
        ),
        (
            lambda loc, covariance: MultivariateNormalFullCovariance(
                loc, covariance
            ),
            lambda loc, covariance: tfd.MultivariateNormalFullCovariance(
                loc=loc, covariance_matrix=covariance
            ),
            (
                as_float([0.1, -0.2]),
                as_float([[1.3, 0.2], [0.2, 0.9]]),
            ),
            jax.random.PRNGKey(17),
            1e-10,
        ),
        (
            lambda loc, scale: Independent(Normal(loc, scale), 1),
            lambda loc, scale: tfd.Independent(tfd.Normal(loc, scale), 1),
            (as_float([0.2, 0.4]), as_float([0.9, 1.1])),
            jax.random.PRNGKey(19),
            1e-10,
        ),
        (
            lambda loc, scale: TransformedDistribution(
                Normal(loc, scale),
                Sigmoid(as_float(-1.0), as_float(2.0)),
            ),
            lambda loc, scale: tfd.TransformedDistribution(
                tfd.Normal(loc, scale),
                tfb.Sigmoid(as_float(-1.0), as_float(2.0)),
            ),
            (as_float(0.1), as_float(0.7)),
            jax.random.PRNGKey(21),
            1e-10,
        ),
    ]
    for build_ours, build_theirs, params, key, atol in cases:
        assert_sample_gradient_parity(
            build_ours, build_theirs, params, key, atol=atol
        )
