import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.prior import Prior
from paz.inference.observable import Observable
from paz.inference.latent import Latent
from paz.inference.pgm import PGM

tfd = tfp.distributions
tfb = tfp.bijectors


# =============================================================================
# Test Fixtures: Model Builders
# =============================================================================


def build_single_prior_model():
    """Model: x ~ Normal(0, 1)"""
    x = Prior("x", tfd.Normal(0.0, 1.0))
    return PGM([x], [x], "single_prior")


def build_single_prior_with_bijector_model():
    """Model: x ~ Uniform(0.1, 1.0) with sigmoid bijector"""
    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    x = Prior("x", tfd.Uniform(low, high), bijector=bijector)
    return PGM([x], [x], "single_prior_bijector"), bijector, (low, high)


def build_three_priors_one_observable_model():
    """Model: Bayesian linear regression
    mean ~ Normal(0, 1)
    bias ~ Normal(0, 1)
    stdv ~ Uniform(0.001, 0.3) with bijector
    y ~ Normal(mean * X + bias, stdv)
    """
    X = jp.linspace(0, 1, 50)
    observations = 0.5 * X + 0.1

    def Likelihood(X):
        def apply(mean, bias, stdv):
            return tfd.Normal(jax.vmap(lambda x: mean * x + bias)(X), stdv)
        return apply

    mean = Prior("mean", tfd.Normal(0.0, 1.0))
    bias = Prior("bias", tfd.Normal(0.0, 1.0))
    low, high = 0.001, 0.3
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    stdv = Prior("stdv", tfd.Uniform(low, high), bijector=bijector)
    y = Observable("y_pred", Likelihood(X), observations)(mean, bias, stdv)
    return PGM([mean, bias, stdv], [y], "linear_regression"), bijector, (low, high)


def build_hierarchical_model():
    """Model: Hierarchical normal
    mu ~ Normal(0, 1)
    sigma ~ Uniform(0.1, 2.0) with bijector
    x ~ Normal(mu, sigma)  [latent]
    y ~ Normal(x, 0.1)     [observable]
    """
    observation = jp.array(1.5)

    low, high = 0.1, 2.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])

    mu = Prior("mu", tfd.Normal(0.0, 1.0))
    sigma = Prior("sigma", tfd.Uniform(low, high), bijector=bijector)

    def x_distribution(mu, sigma):
        return tfd.Normal(mu, sigma)

    x = Latent("x", x_distribution)(mu, sigma)

    def y_distribution(x):
        return tfd.Normal(x, 0.1)

    y = Observable("y", y_distribution, observation)(x)
    return PGM([mu, sigma], [y], "hierarchical"), bijector, (low, high)


def build_two_observables_model():
    """Model: Two observables sharing priors
    mean ~ Normal(0, 1)
    stdv ~ Uniform(0.1, 1.0) with bijector
    y1 ~ Normal(mean, stdv)
    y2 ~ Normal(mean + 1, stdv)
    """
    obs1 = jp.array([0.5, 0.6, 0.7])
    obs2 = jp.array([1.5, 1.6, 1.7])

    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])

    mean = Prior("mean", tfd.Normal(0.0, 1.0))
    stdv = Prior("stdv", tfd.Uniform(low, high), bijector=bijector)

    def likelihood1(mean, stdv):
        return tfd.Normal(mean, stdv)

    def likelihood2(mean, stdv):
        return tfd.Normal(mean + 1, stdv)

    y1 = Observable("y1", likelihood1, obs1)(mean, stdv)
    y2 = Observable("y2", likelihood2, obs2)(mean, stdv)
    return PGM([mean, stdv], [y1, y2], "two_observables"), bijector, (low, high)


def build_vector_latent_model(num_groups):
    """Model: Vector latent depends on scalar parents."""
    observation = jp.zeros((num_groups,))

    mu = Prior("mu", tfd.Normal(0.0, 1.0))
    sigma = Prior("sigma", tfd.Normal(1.0, 0.2))

    def x_distribution(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), sigma),
            reinterpreted_batch_ndims=1,
        )

    x = Latent("x", x_distribution)(mu, sigma)

    def y_distribution(x):
        return tfd.Normal(x, 1.0)

    y = Observable("y", y_distribution, observation)(x)
    return PGM([mu, sigma], [y], "vector_latent")


def build_multi_parent_observable_model():
    """Model: Observable depends on two latent parents."""
    observation = jp.array(0.0)

    mu1 = Prior("mu1", tfd.Normal(0.0, 1.0))
    mu2 = Prior("mu2", tfd.Normal(0.0, 1.0))

    def x_distribution(mu1):
        return tfd.Normal(mu1, 1.0)

    def z_distribution(mu2):
        return tfd.Normal(mu2, 1.0)

    x = Latent("x", x_distribution)(mu1)
    z = Latent("z", z_distribution)(mu2)

    def y_distribution(x, z):
        return tfd.Normal(x + z, 1.0)

    y = Observable("y", y_distribution, observation)(x, z)
    return PGM([mu1, mu2], [y], "multi_parent")


# =============================================================================
# Tests: sample_inverse shapes
# =============================================================================


def test_sample_inverse_single_prior_single_sample_shape():
    model = build_single_prior_model()
    key = jax.random.PRNGKey(0)
    samples = model.sample_inverse(key, num_samples=1)
    assert samples.x.shape == (), f"Expected scalar, got {samples.x.shape}"


def test_sample_inverse_single_prior_multiple_samples_shape():
    model = build_single_prior_model()
    key = jax.random.PRNGKey(0)
    for n in [2, 4, 10]:
        samples = model.sample_inverse(key, num_samples=n)
        assert samples.x.shape == (n,), f"Expected ({n},), got {samples.x.shape}"


def test_sample_inverse_three_priors_single_sample_shape():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)
    samples = model.sample_inverse(key, num_samples=1)
    assert samples.mean.shape == (), f"Expected scalar, got {samples.mean.shape}"
    assert samples.bias.shape == (), f"Expected scalar, got {samples.bias.shape}"
    assert samples.stdv.shape == (), f"Expected scalar, got {samples.stdv.shape}"


def test_sample_inverse_three_priors_multiple_samples_shape():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)
    for n in [2, 4, 10]:
        samples = model.sample_inverse(key, num_samples=n)
        assert samples.mean.shape == (n,), f"mean: expected ({n},), got {samples.mean.shape}"
        assert samples.bias.shape == (n,), f"bias: expected ({n},), got {samples.bias.shape}"
        assert samples.stdv.shape == (n,), f"stdv: expected ({n},), got {samples.stdv.shape}"


def test_sample_inverse_hierarchical_model_shape():
    model, _, _ = build_hierarchical_model()
    key = jax.random.PRNGKey(0)
    for n in [1, 4]:
        samples = model.sample_inverse(key, num_samples=n)
        expected = () if n == 1 else (n,)
        assert samples.mu.shape == expected, f"mu: expected {expected}, got {samples.mu.shape}"
        assert samples.sigma.shape == expected, f"sigma: expected {expected}, got {samples.sigma.shape}"
        assert samples.x.shape == expected, f"x: expected {expected}, got {samples.x.shape}"


# =============================================================================
# Tests: sample_inverse returns only latent variables (not observables)
# =============================================================================


def test_sample_inverse_excludes_observables():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)
    samples = model.sample_inverse(key)
    assert hasattr(samples, "mean")
    assert hasattr(samples, "bias")
    assert hasattr(samples, "stdv")
    assert not hasattr(samples, "y_pred"), "sample_inverse should not include observables"


def test_sample_inverse_hierarchical_includes_latent():
    model, _, _ = build_hierarchical_model()
    key = jax.random.PRNGKey(0)
    samples = model.sample_inverse(key)
    assert hasattr(samples, "mu")
    assert hasattr(samples, "sigma")
    assert hasattr(samples, "x"), "sample_inverse should include latent variable x"
    assert not hasattr(samples, "y"), "sample_inverse should not include observable y"


# =============================================================================
# Tests: sample_inverse returns unconstrained values
# =============================================================================


def test_sample_inverse_returns_unconstrained_values():
    model, bijector, (low, high) = build_single_prior_with_bijector_model()
    key = jax.random.PRNGKey(42)
    samples = model.sample_inverse(key, num_samples=1000)
    assert samples.x.min() < low, "Unconstrained samples should go below low bound"
    assert samples.x.max() > high, "Unconstrained samples should go above high bound"


def test_sample_inverse_stdv_unconstrained():
    model, bijector, (low, high) = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(42)
    samples = model.sample_inverse(key, num_samples=1000)
    assert samples.stdv.min() < low, "Unconstrained stdv should go below low bound"
    assert samples.stdv.max() > high, "Unconstrained stdv should go above high bound"


# =============================================================================
# Tests: sample_forward shapes and constrained values
# =============================================================================


def test_sample_forward_single_prior_shape():
    model = build_single_prior_model()
    key = jax.random.PRNGKey(0)
    samples = model.sample(key, num_samples=1)
    assert samples.x.shape == ()
    samples = model.sample(key, num_samples=4)
    assert samples.x.shape == (4,)


def test_sample_forward_three_priors_includes_observable():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)
    samples = model.sample(key)
    assert hasattr(samples, "mean")
    assert hasattr(samples, "bias")
    assert hasattr(samples, "stdv")
    assert hasattr(samples, "y_pred"), "sample_forward should include observables"


def test_sample_forward_returns_constrained_values():
    model, bijector, (low, high) = build_single_prior_with_bijector_model()
    key = jax.random.PRNGKey(42)
    samples = model.sample(key, num_samples=1000)
    assert samples.x.min() >= low, f"Constrained samples should be >= {low}"
    assert samples.x.max() <= high, f"Constrained samples should be <= {high}"


def test_sample_forward_stdv_constrained():
    model, bijector, (low, high) = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(42)
    samples = model.sample(key, num_samples=1000)
    assert samples.stdv.min() >= low, f"Constrained stdv should be >= {low}"
    assert samples.stdv.max() <= high, f"Constrained stdv should be <= {high}"


# =============================================================================
# Tests: apply returns correct structure
# =============================================================================


def test_apply_returns_node_state():
    model = build_single_prior_model()
    key = jax.random.PRNGKey(0)
    params = model.sample_inverse(key)
    state = model.apply(params)
    assert hasattr(state, "sample")
    assert hasattr(state, "log_prob")
    assert hasattr(state, "log_prob_sum")


def test_apply_log_prob_is_dict():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)
    params = model.sample_inverse(key)
    state = model.apply(params)
    assert isinstance(state.log_prob, dict)
    assert "mean" in state.log_prob
    assert "bias" in state.log_prob
    assert "stdv" in state.log_prob
    assert "y_pred" in state.log_prob


def test_apply_log_prob_sum_equals_sum_of_log_probs():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)
    params = model.sample_inverse(key)
    state = model.apply(params)
    expected_sum = sum(state.log_prob.values())
    assert jp.isclose(state.log_prob_sum, expected_sum), \
        f"log_prob_sum {state.log_prob_sum} != sum {expected_sum}"


# =============================================================================
# Tests: apply returns constrained samples
# =============================================================================


def test_apply_returns_constrained_sample():
    model, bijector, (low, high) = build_single_prior_with_bijector_model()
    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        state = model.apply(params)
        assert low <= state.sample.x <= high, \
            f"apply sample {state.sample.x} not in [{low}, {high}]"


def test_apply_returns_constrained_stdv():
    model, bijector, (low, high) = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        state = model.apply(params)
        assert low <= state.sample.stdv <= high, \
            f"apply stdv {state.sample.stdv} not in [{low}, {high}]"


# =============================================================================
# Tests: apply produces finite log_prob (no NaN/inf) - regression test for bug
# =============================================================================


def test_apply_no_nan_single_prior_with_bijector():
    model, _, _ = build_single_prior_with_bijector_model()
    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        state = model.apply(params)
        assert jp.isfinite(state.log_prob_sum), \
            f"log_prob_sum is not finite: {state.log_prob_sum}"


def test_apply_no_nan_three_priors():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        state = model.apply(params)
        assert jp.isfinite(state.log_prob_sum), \
            f"log_prob_sum is not finite: {state.log_prob_sum}, params={params}"


def test_apply_no_nan_hierarchical():
    model, _, _ = build_hierarchical_model()
    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        state = model.apply(params)
        assert jp.isfinite(state.log_prob_sum), \
            f"log_prob_sum is not finite: {state.log_prob_sum}, params={params}"


def test_apply_no_nan_two_observables():
    model, _, _ = build_two_observables_model()
    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        state = model.apply(params)
        assert jp.isfinite(state.log_prob_sum), \
            f"log_prob_sum is not finite: {state.log_prob_sum}, params={params}"


def test_apply_no_nan_multi_parent_observable():
    model = build_multi_parent_observable_model()
    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        state = model.apply(params)
        assert jp.isfinite(state.log_prob_sum), \
            f"log_prob_sum is not finite: {state.log_prob_sum}"


# =============================================================================
# Tests: vmap over apply works (required for MCMC)
# =============================================================================


def test_vmap_apply_single_prior():
    model = build_single_prior_model()
    key = jax.random.PRNGKey(0)
    positions = model.sample_inverse(key, num_samples=4)
    log_density_fn = lambda params: model.apply(params).log_prob_sum
    log_densities = jax.vmap(log_density_fn)(positions)
    assert log_densities.shape == (4,)
    assert jp.all(jp.isfinite(log_densities))


def test_vmap_apply_three_priors():
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)
    positions = model.sample_inverse(key, num_samples=4)
    log_density_fn = lambda params: model.apply(params).log_prob_sum
    log_densities = jax.vmap(log_density_fn)(positions)
    assert log_densities.shape == (4,)
    assert jp.all(jp.isfinite(log_densities)), f"NaN in log_densities: {log_densities}"


def test_vmap_apply_hierarchical():
    model, _, _ = build_hierarchical_model()
    key = jax.random.PRNGKey(0)
    positions = model.sample_inverse(key, num_samples=4)
    log_density_fn = lambda params: model.apply(params).log_prob_sum
    log_densities = jax.vmap(log_density_fn)(positions)
    assert log_densities.shape == (4,)
    assert jp.all(jp.isfinite(log_densities)), f"NaN in log_densities: {log_densities}"


# =============================================================================
# Tests: Bijector Jacobian correction is applied
# =============================================================================


def test_apply_includes_jacobian_correction():
    """Verify that log_prob includes the Jacobian correction from bijector."""
    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    distribution = tfd.Uniform(low, high)
    x = Prior("x", distribution, bijector=bijector)
    model = PGM([x], [x], "test")

    key = jax.random.PRNGKey(42)
    unconstrained = model.sample_inverse(key)
    state = model.apply(unconstrained)

    constrained = bijector(unconstrained.x)
    log_prob_without_jacobian = distribution.log_prob(constrained)
    jacobian = bijector.forward_log_det_jacobian(unconstrained.x)
    log_prob_with_jacobian = log_prob_without_jacobian + jacobian

    assert jp.isclose(state.log_prob["x"], log_prob_with_jacobian), \
        f"Jacobian not applied: {state.log_prob['x']} != {log_prob_with_jacobian}"


# =============================================================================
# Tests: Observable receives constrained parameters - regression test for bug
# =============================================================================


def test_observable_receives_constrained_stdv():
    """
    Regression test: Observable must receive constrained (positive) stdv,
    not unconstrained (possibly negative) value.
    """
    X = jp.array([0.0, 1.0])
    observations = jp.array([0.1, 0.6])

    def Likelihood(X):
        def apply(stdv):
            return tfd.Normal(X, stdv)
        return apply

    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    stdv = Prior("stdv", tfd.Uniform(low, high), bijector=bijector)
    y = Observable("y", Likelihood(X), observations)(stdv)
    model = PGM([stdv], [y], "test")

    key = jax.random.PRNGKey(42)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        params = model.sample_inverse(subkey)
        if params.stdv < 0:
            state = model.apply(params)
            assert jp.isfinite(state.log_prob_sum), \
                f"NaN when unconstrained stdv={params.stdv} (should be transformed)"


def test_observable_scale_is_positive():
    """Verify that Normal distribution in observable has positive scale."""
    X = jp.array([0.0, 1.0])
    observations = jp.array([0.1, 0.6])

    def Likelihood(X):
        def apply(stdv):
            return tfd.Normal(X, stdv)
        return apply

    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    stdv = Prior("stdv", tfd.Uniform(low, high), bijector=bijector)
    y = Observable("y", Likelihood(X), observations)(stdv)
    model = PGM([stdv], [y], "test")

    key = jax.random.PRNGKey(42)
    params = model.sample_inverse(key)
    state = model.apply(params)
    distribution = state.sample.y
    assert distribution.scale > 0, f"Observable scale should be positive, got {distribution.scale}"


# =============================================================================
# Tests: sample_inverse preserves batch dimension - regression test for bug
# =============================================================================


def test_sample_inverse_batch_not_overwritten():
    """
    Regression test: sample_inverse with num_samples > 1 should return
    arrays with shape (num_samples,), not scalars.
    """
    model, _, _ = build_three_priors_one_observable_model()
    key = jax.random.PRNGKey(0)

    for n in [2, 4, 8]:
        samples = model.sample_inverse(key, num_samples=n)
        assert samples.mean.shape == (n,), \
            f"mean shape {samples.mean.shape} != ({n},) - batch was overwritten"
        assert samples.bias.shape == (n,), \
            f"bias shape {samples.bias.shape} != ({n},) - batch was overwritten"
        assert samples.stdv.shape == (n,), \
            f"stdv shape {samples.stdv.shape} != ({n},) - batch was overwritten"


def test_sample_inverse_hierarchical_batch_preserved():
    """
    Regression test for hierarchical model: latent variable x should also
    have correct batch dimension.
    """
    model, _, _ = build_hierarchical_model()
    key = jax.random.PRNGKey(0)

    for n in [2, 4, 8]:
        samples = model.sample_inverse(key, num_samples=n)
        assert samples.mu.shape == (n,), f"mu shape {samples.mu.shape} != ({n},)"
        assert samples.sigma.shape == (n,), f"sigma shape {samples.sigma.shape} != ({n},)"
        assert samples.x.shape == (n,), f"x shape {samples.x.shape} != ({n},)"


def test_sample_inverse_multi_parent_batch_preserved():
    model = build_multi_parent_observable_model()
    key = jax.random.PRNGKey(0)
    for n in [2, 4, 8]:
        samples = model.sample_inverse(key, num_samples=n)
        assert samples.mu1.shape == (n,), f"mu1 shape {samples.mu1.shape} != ({n},)"
        assert samples.mu2.shape == (n,), f"mu2 shape {samples.mu2.shape} != ({n},)"
        assert samples.x.shape == (n,), f"x shape {samples.x.shape} != ({n},)"
        assert samples.z.shape == (n,), f"z shape {samples.z.shape} != ({n},)"


def test_sample_inverse_vector_latent_batch_preserved():
    model = build_vector_latent_model(num_groups=5)
    key = jax.random.PRNGKey(0)
    for n in [2, 4, 8]:
        samples = model.sample_inverse(key, num_samples=n)
        assert samples.mu.shape == (n,), f"mu shape {samples.mu.shape} != ({n},)"
        assert samples.sigma.shape == (n,), f"sigma shape {samples.sigma.shape} != ({n},)"
        assert samples.x.shape == (n, 5), (
            f"x shape {samples.x.shape} != ({n}, 5)"
        )


def test_sample_inverse_vector_latent_single_sample_shape():
    model = build_vector_latent_model(num_groups=3)
    key = jax.random.PRNGKey(0)
    samples = model.sample_inverse(key, num_samples=1)
    assert samples.mu.shape == (), f"mu shape {samples.mu.shape} != ()"
    assert samples.sigma.shape == (), f"sigma shape {samples.sigma.shape} != ()"
    assert samples.x.shape == (3,), f"x shape {samples.x.shape} != (3,)"


# =============================================================================
# Tests: Different random keys produce different samples
# =============================================================================


def test_different_keys_different_samples():
    model = build_single_prior_model()
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)
    samples1 = model.sample_inverse(key1)
    samples2 = model.sample_inverse(key2)
    assert samples1.x != samples2.x, "Different keys should produce different samples"


def test_same_key_same_samples():
    model = build_single_prior_model()
    key = jax.random.PRNGKey(42)
    samples1 = model.sample_inverse(key)
    samples2 = model.sample_inverse(key)
    assert samples1.x == samples2.x, "Same key should produce same samples"


# =============================================================================
# Tests: Model with multiple observables
# =============================================================================


def test_two_observables_sample_forward_shape():
    model, _, _ = build_two_observables_model()
    key = jax.random.PRNGKey(0)
    samples = model.sample(key)
    assert hasattr(samples, "mean")
    assert hasattr(samples, "stdv")
    assert hasattr(samples, "y1")
    assert hasattr(samples, "y2")


def test_two_observables_apply_log_prob():
    model, _, _ = build_two_observables_model()
    key = jax.random.PRNGKey(0)
    params = model.sample_inverse(key)
    state = model.apply(params)
    assert "y1" in state.log_prob
    assert "y2" in state.log_prob
    assert jp.isfinite(state.log_prob["y1"])
    assert jp.isfinite(state.log_prob["y2"])


# =============================================================================
# Tests: Edge cases
# =============================================================================


def test_extreme_unconstrained_values():
    """Test that extreme unconstrained values don't cause NaN."""
    model, bijector, (low, high) = build_single_prior_with_bijector_model()
    from collections import namedtuple
    Sample = namedtuple("Sample", ["x"])

    extreme_values = [-10.0, -5.0, 0.0, 5.0, 10.0]
    for val in extreme_values:
        params = Sample(x=jp.array(val))
        state = model.apply(params)
        assert jp.isfinite(state.log_prob_sum), \
            f"NaN for extreme unconstrained value {val}"
        assert low <= state.sample.x <= high, \
            f"Constrained value {state.sample.x} out of bounds for input {val}"


def test_apply_with_manual_params():
    """Test apply with manually constructed parameters."""
    model, _, _ = build_three_priors_one_observable_model()
    from collections import namedtuple
    Sample = namedtuple("Sample", ["mean", "bias", "stdv"])

    params = Sample(mean=jp.array(0.5), bias=jp.array(0.1), stdv=jp.array(0.0))
    state = model.apply(params)
    assert jp.isfinite(state.log_prob_sum)
    assert state.sample.stdv > 0, "stdv should be transformed to positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
