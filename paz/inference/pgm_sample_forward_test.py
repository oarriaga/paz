import jax
import jax.numpy as jp
import paz
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def test_sample_forward_simple_model():
    """Test sample_forward with a simple model (no Latent nodes)."""
    mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
    stdv = paz.Prior(tfd.Normal(0.0, 1.0), name="stdv")

    def likelihood(mu, sigma):
        return tfd.Normal(mu, jp.abs(sigma))

    y_obs = paz.Observable(likelihood, name="y_obs")(mean, stdv)
    model = paz.PGM([mean, stdv], [y_obs], "simple")

    key = jax.random.PRNGKey(0)
    sample = model.sample(key, num_samples=1)

    assert hasattr(sample, "mean")
    assert hasattr(sample, "stdv")
    assert hasattr(sample, "y_obs")
    assert sample.mean.shape == ()
    assert sample.stdv.shape == ()
    assert sample.y_obs.shape == ()
    print("✓ test_sample_forward_simple_model passed")


def test_sample_forward_simple_model_batched():
    """Test sample_forward with batched sampling (no Latent nodes)."""
    mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
    stdv = paz.Prior(tfd.Normal(0.0, 1.0), name="stdv")

    def likelihood(mu, sigma):
        return tfd.Normal(mu, jp.abs(sigma))

    y_obs = paz.Observable(likelihood, name="y_obs")(mean, stdv)
    model = paz.PGM([mean, stdv], [y_obs], "simple")

    key = jax.random.PRNGKey(0)
    num_samples = 10
    sample = model.sample(key, num_samples=num_samples)

    print(f"  mean shape: {sample.mean.shape}, expected: ({num_samples},)")
    print(f"  stdv shape: {sample.stdv.shape}, expected: ({num_samples},)")
    print(f"  y_obs shape: {sample.y_obs.shape}, expected: ({num_samples},)")

    assert sample.mean.shape == (num_samples,)
    assert sample.stdv.shape == (num_samples,)
    assert sample.y_obs.shape == (num_samples,)
    print("✓ test_sample_forward_simple_model_batched passed")


def test_sample_forward_hierarchical_single_sample():
    """Test sample_forward with Latent nodes (single sample)."""
    num_groups = 3

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), sigma),
            reinterpreted_batch_ndims=1
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "hierarchical")

    key = jax.random.PRNGKey(0)
    sample = model.sample(key, num_samples=1)

    assert hasattr(sample, "mu")
    assert hasattr(sample, "sigma")
    assert hasattr(sample, "groups")
    assert hasattr(sample, "y_obs")
    assert sample.mu.shape == ()
    assert sample.sigma.shape == ()
    assert sample.groups.shape == (num_groups,)
    assert sample.y_obs.shape == (num_groups,)
    print("✓ test_sample_forward_hierarchical_single_sample passed")


def test_sample_forward_hierarchical_batched():
    """Test sample_forward with Latent nodes (batched sampling)."""
    num_groups = 3
    num_samples = 10

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), sigma),
            reinterpreted_batch_ndims=1
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "hierarchical")

    key = jax.random.PRNGKey(0)
    sample = model.sample(key, num_samples=num_samples)

    print(f"mu shape: {sample.mu.shape}")
    print(f"sigma shape: {sample.sigma.shape}")
    print(f"groups shape: {sample.groups.shape}")
    print(f"y_obs shape: {sample.y_obs.shape}")

    assert sample.mu.shape == (num_samples,), f"Expected (10,), got {sample.mu.shape}"
    assert sample.sigma.shape == (num_samples,), f"Expected (10,), got {sample.sigma.shape}"
    assert sample.groups.shape == (num_samples, num_groups), \
        f"Expected ({num_samples}, {num_groups}), got {sample.groups.shape}"
    assert sample.y_obs.shape == (num_samples, num_groups), \
        f"Expected ({num_samples}, {num_groups}), got {sample.y_obs.shape}"

    for i in range(num_samples):
        group_mean = sample.mu[i]
        expected_groups = tfd.Normal(jp.full(num_groups, group_mean), jp.abs(sample.sigma[i]))
        actual_groups = sample.groups[i]
        print(f"Sample {i}: mu={group_mean:.3f}, groups={actual_groups}")

    print("✓ test_sample_forward_hierarchical_batched passed")


def test_sample_forward_hierarchical_regression():
    """Test sample_forward with hierarchical regression structure."""
    num_groups = 5
    num_samples = 10

    def slope_prior(mu_slope, sigma_slope):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_slope), sigma_slope),
            reinterpreted_batch_ndims=1
        )

    def intercept_prior(mu_intercept, sigma_intercept):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_intercept), sigma_intercept),
            reinterpreted_batch_ndims=1
        )

    mu_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_slope")
    mu_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_intercept")
    sigma_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma_slope")
    sigma_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma_intercept")

    slopes = paz.Latent(slope_prior, name="slopes")(mu_slope, sigma_slope)
    intercepts = paz.Latent(intercept_prior, name="intercepts")(
        mu_intercept, sigma_intercept
    )

    X = jp.linspace(0, 1, 50)
    group_idx = jp.repeat(jp.arange(num_groups), 10)

    def likelihood(slopes, intercepts):
        means = slopes[group_idx] * X + intercepts[group_idx]
        return tfd.Normal(means, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(
        slopes, intercepts
    )

    priors = [mu_slope, mu_intercept, sigma_slope, sigma_intercept]
    model = paz.PGM(priors, [y_obs], "hierarchical_regression")

    key = jax.random.PRNGKey(42)
    sample = model.sample(key, num_samples=num_samples)

    print(f"mu_slope shape: {sample.mu_slope.shape}")
    print(f"mu_intercept shape: {sample.mu_intercept.shape}")
    print(f"slopes shape: {sample.slopes.shape}")
    print(f"intercepts shape: {sample.intercepts.shape}")
    print(f"y_obs shape: {sample.y_obs.shape}")

    assert sample.mu_slope.shape == (num_samples,)
    assert sample.mu_intercept.shape == (num_samples,)
    assert sample.slopes.shape == (num_samples, num_groups), \
        f"Expected ({num_samples}, {num_groups}), got {sample.slopes.shape}"
    assert sample.intercepts.shape == (num_samples, num_groups), \
        f"Expected ({num_samples}, {num_groups}), got {sample.intercepts.shape}"
    assert sample.y_obs.shape == (num_samples, 50)

    print("✓ test_sample_forward_hierarchical_regression passed")


if __name__ == "__main__":
    print("Running sample_forward tests...\n")
    test_sample_forward_simple_model()
    test_sample_forward_simple_model_batched()
    test_sample_forward_hierarchical_single_sample()
    print("\nTesting hierarchical batched sampling (this may fail)...")
    try:
        test_sample_forward_hierarchical_batched()
    except AssertionError as e:
        print(f"✗ test_sample_forward_hierarchical_batched failed: {e}")
    print("\nTesting hierarchical regression sampling (this may fail)...")
    try:
        test_sample_forward_hierarchical_regression()
    except Exception as e:
        print(f"✗ test_sample_forward_hierarchical_regression failed: {e}")
    print("\nDone!")
