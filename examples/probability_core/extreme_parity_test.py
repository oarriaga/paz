import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from .bijectors import Exp, Sigmoid, SoftmaxCentered, Softplus
from .distributions import (
    Beta,
    Bernoulli,
    Categorical,
    LogNormal,
    Normal,
    Poisson,
    QuantizedDistribution,
    RelaxedOneHotCategorical,
    TransformedDistribution,
    TruncatedNormal,
    VonMises,
)
from .parity_helpers import (
    assert_exact_method_parity,
    assert_gradient_parity,
)
from .bijectors import Chain, Scale, Shift

tfd = tfp.distributions
tfb = tfp.bijectors


def as_float(values):
    return jp.asarray(values, dtype=jp.float64)


def test_normal_extreme_masks_match_tfp():
    values = as_float([-jp.inf, -1e300, 0.0, 1e300, jp.inf, jp.nan])
    ours = Normal(as_float(0.0), as_float(1.0))
    theirs = tfd.Normal(as_float(0.0), as_float(1.0))
    assert_exact_method_parity(ours, theirs, "log_prob", values)
    assert_exact_method_parity(ours, theirs, "cdf", values)
    assert_exact_method_parity(ours, theirs, "log_cdf", values)


def test_log_normal_zero_and_inf_masks_match_tfp():
    tiny = jp.finfo(jp.float64).tiny
    values = as_float([0.0, tiny, 1.0, jp.inf, jp.nan])
    ours = LogNormal(as_float(0.0), as_float(1.0))
    theirs = tfd.LogNormal(as_float(0.0), as_float(1.0))
    assert_exact_method_parity(ours, theirs, "log_prob", values)
    assert_exact_method_parity(ours, theirs, "cdf", values)
    assert_exact_method_parity(ours, theirs, "log_cdf", values)


def test_truncated_normal_boundary_masks_match_tfp():
    values = as_float([-jp.inf, -1.0, 0.0, 1.0, jp.inf, jp.nan])
    ours = TruncatedNormal(
        as_float(0.0), as_float(1.0), as_float(-1.0), as_float(1.0)
    )
    theirs = tfd.TruncatedNormal(
        as_float(0.0), as_float(1.0), as_float(-1.0), as_float(1.0)
    )
    assert_exact_method_parity(ours, theirs, "log_prob", values)
    assert_exact_method_parity(ours, theirs, "cdf", values)
    assert_exact_method_parity(ours, theirs, "log_cdf", values)


def test_beta_boundary_masks_match_tfp():
    tiny = jp.finfo(jp.float64).tiny
    values = as_float([0.0, tiny, 0.5, 1.0 - tiny, 1.0, -0.1, 1.1, jp.nan])
    ours = Beta(as_float(0.5), as_float(0.5))
    theirs = tfd.Beta(as_float(0.5), as_float(0.5))
    assert_exact_method_parity(ours, theirs, "log_prob", values)


def test_von_mises_large_concentration_masks_match_tfp():
    values = as_float([-jp.pi, 0.0, jp.pi, jp.nan])
    ours = VonMises(as_float(0.0), as_float(1000.0))
    theirs = tfd.VonMises(as_float(0.0), as_float(1000.0))
    assert_exact_method_parity(ours, theirs, "log_prob", values)


def test_discrete_extreme_masks_match_tfp():
    bernoulli_ours = Bernoulli(probs=as_float([0.0, 1.0]), dtype=jp.float64)
    bernoulli_theirs = tfd.Bernoulli(
        probs=as_float([0.0, 1.0]), dtype=jp.float64
    )
    assert_exact_method_parity(
        bernoulli_ours, bernoulli_theirs, "log_prob", as_float([0.0, 1.0])
    )

    categorical_ours = Categorical(
        probs=as_float([0.0, 1.0, 0.0]),
        force_probs_to_zero_outside_support=True,
    )
    categorical_theirs = tfd.Categorical(
        probs=as_float([0.0, 1.0, 0.0]),
        force_probs_to_zero_outside_support=True,
    )
    assert_exact_method_parity(
        categorical_ours,
        categorical_theirs,
        "log_prob",
        jp.array([-1, 0, 1, 2, 3]),
    )

    poisson_ours = Poisson(
        rate=as_float(0.0), force_probs_to_zero_outside_support=True
    )
    poisson_theirs = tfd.Poisson(
        rate=as_float(0.0), force_probs_to_zero_outside_support=True
    )
    assert_exact_method_parity(
        poisson_ours,
        poisson_theirs,
        "log_prob",
        as_float([0.0, 1.0, -1.0, 2.5, jp.nan]),
    )


def test_relaxed_and_quantized_extreme_masks_match_tfp():
    tiny = jp.finfo(jp.float64).tiny
    relaxed_ours = RelaxedOneHotCategorical(
        as_float(0.7), probs=as_float([0.2, 0.5, 0.3])
    )
    relaxed_theirs = tfd.RelaxedOneHotCategorical(
        as_float(0.7), probs=as_float([0.2, 0.5, 0.3])
    )
    values = as_float([[tiny, 0.7, 0.3], [0.0, 1.0, 0.0]])
    assert_exact_method_parity(
        relaxed_ours, relaxed_theirs, "log_prob", values
    )

    quantized_ours = QuantizedDistribution(
        TransformedDistribution(
            Normal(as_float(0.0), as_float(1.0)),
            Chain([
                Shift(as_float(-0.5)),
                Shift(as_float(1.5)),
                Scale(as_float(2.0)),
            ]),
        )
    )
    quantized_theirs = tfd.QuantizedDistribution(
        tfd.TransformedDistribution(
            tfd.Normal(as_float(0.0), as_float(1.0)),
            tfb.Chain(
                [
                    tfb.Shift(as_float(-0.5)),
                    tfb.Shift(as_float(1.5)),
                    tfb.Scale(as_float(2.0)),
                ]
            ),
        )
    )
    assert_exact_method_parity(
        quantized_ours,
        quantized_theirs,
        "log_prob",
        as_float([-jp.inf, -1.0, 0.0, 2.0, jp.nan]),
    )


def test_sigmoid_inverse_extreme_masks_match_tfp():
    tiny = jp.finfo(jp.float64).tiny
    values = as_float([0.0, tiny, 0.5, 1.0 - tiny, 1.0, jp.nan])
    ours = Sigmoid()
    theirs = tfb.Sigmoid()
    assert_exact_method_parity(ours, theirs, "inverse", values)

    bounded_values = as_float(
        [-2.0, -2.0 + tiny, 0.3, 3.0 - tiny, 3.0, jp.nan]
    )
    bounded_ours = Sigmoid(as_float(-2.0), as_float(3.0))
    bounded_theirs = tfb.Sigmoid(as_float(-2.0), as_float(3.0))
    assert_exact_method_parity(bounded_ours, bounded_theirs, "inverse",
                               bounded_values)


def test_softplus_inverse_extreme_masks_match_tfp():
    tiny = jp.finfo(jp.float64).tiny
    values = as_float([0.0, tiny, 1.0, 1e300, jp.inf, jp.nan])
    ours = Softplus()
    theirs = tfb.Softplus()
    assert_exact_method_parity(ours, theirs, "inverse", values)


def test_exp_and_softmax_centered_extreme_masks_match_tfp():
    tiny = jp.finfo(jp.float64).tiny
    exp_ours = Exp()
    exp_theirs = tfb.Exp()
    assert_exact_method_parity(
        exp_ours,
        exp_theirs,
        "inverse",
        as_float([0.0, tiny, 1.0, jp.inf, jp.nan]),
    )

    softmax_ours = SoftmaxCentered()
    softmax_theirs = tfb.SoftmaxCentered()
    simplex = as_float([tiny, 0.7, 0.3, 0.0])
    assert_exact_method_parity(softmax_ours, softmax_theirs, "inverse", simplex)


def test_sigmoid_inverse_extreme_gradient_parity():
    tiny = jp.finfo(jp.float64).tiny

    def our_loss(values):
        return jp.sum(Sigmoid().inverse(values))

    def their_loss(values):
        return jp.sum(tfb.Sigmoid().inverse(values))

    assert_gradient_parity(
        our_loss, their_loss, (as_float([tiny, 0.5, 1.0 - tiny]),)
    )


def test_softplus_inverse_extreme_gradient_parity():
    tiny = jp.finfo(jp.float64).tiny

    def our_loss(values):
        return jp.sum(Softplus().inverse(values))

    def their_loss(values):
        return jp.sum(tfb.Softplus().inverse(values))

    assert_gradient_parity(
        our_loss, their_loss, (as_float([tiny, 1.0, 1e300]),)
    )


def test_log_normal_zero_gradient_parity():
    tiny = jp.finfo(jp.float64).tiny

    def our_loss(values):
        return jp.sum(LogNormal(as_float(0.0), as_float(1.0)).log_prob(values))

    def their_loss(values):
        return jp.sum(
            tfd.LogNormal(as_float(0.0), as_float(1.0)).log_prob(values)
        )

    assert_gradient_parity(
        our_loss, their_loss, (as_float([tiny, 1.0]),)
    )


def test_beta_boundary_gradient_parity():
    tiny = jp.finfo(jp.float64).tiny

    def our_loss(values):
        return jp.sum(Beta(as_float(0.5), as_float(0.5)).log_prob(values))

    def their_loss(values):
        return jp.sum(
            tfd.Beta(as_float(0.5), as_float(0.5)).log_prob(values)
        )

    assert_gradient_parity(
        our_loss, their_loss, (as_float([tiny, 0.5, 1.0 - tiny]),)
    )
