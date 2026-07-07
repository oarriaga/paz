import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.gated import Gated

tfd = tfp.distributions

PROBS = jp.array([0.25, 0.5, 0.25])


def build_choice():
    return tfd.Categorical(probs=PROBS, dtype=jp.float32)


def test_gated_invalid_callable_raises():
    with pytest.raises(ValueError):
        Gated("not-callable")


def test_gated_off_samples_zero_with_zero_log_prob():
    gated = Gated(build_choice, name="x")()
    sample = gated.sample(jax.random.PRNGKey(0), 1, jp.array(0.0))
    state = gated.log_prob(sample, jp.array(0.0))
    assert sample == 0.0
    assert jp.isclose(state.log_prob_sum, 0.0)


def test_gated_on_matches_inner_distribution():
    gated = Gated(build_choice, name="x")()
    state = gated.log_prob(jp.array(2.0), jp.array(1.0))
    expected = build_choice().log_prob(jp.array(2.0))
    assert jp.isclose(state.log_prob_sum, expected)


def test_gated_inside_pgm_scores_both_branches():
    include = paz.Prior(tfd.Bernoulli(probs=0.5, dtype=jp.float32),
                        name="include")
    choice = Gated(build_choice, name="choice")(include)
    model = paz.PGM([include], [include, choice], "gated")
    for seed in range(4):
        values = model.sample(jax.random.PRNGKey(seed))
        expected = jp.log(0.5)
        if values.include == 1.0:
            expected = expected + build_choice().log_prob(values.choice)
        state = model.prior.log_prob(values)
        assert jp.isclose(state.log_prob_sum, expected)
