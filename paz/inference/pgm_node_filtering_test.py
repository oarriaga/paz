import jax
import jax.numpy as jp
import paz
from tensorflow_probability.substrates import jax as tfp
from paz.inference.pgm import get_latent_nodes, get_non_root_nodes, search_nodes, get_edges
from paz.abstract.dag import DAG

tfd = tfp.distributions


def test_latent_nodes_includes_priors():
    """Verify that get_latent_nodes returns nodes with sample_inverse, including Priors."""
    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")

    def latent_prior(mu_val, sigma_val):
        return tfd.Normal(mu_val, jp.abs(sigma_val))

    latent = paz.Latent(latent_prior, name="latent")(mu, sigma)

    def likelihood(latent_val):
        return tfd.Normal(latent_val, 0.1)

    y_obs = paz.Observable(likelihood, 0.0, name="y_obs")(latent)

    nodes = search_nodes([y_obs])
    latent_nodes = get_latent_nodes(nodes)
    latent_names = [node.name for node in latent_nodes]

    print(f"All nodes: {[n.name for n in nodes]}")
    print(f"Latent nodes (with sample_inverse): {latent_names}")

    assert "mu" in latent_names, "mu (Prior) should be in latent_nodes"
    assert "sigma" in latent_names, "sigma (Prior) should be in latent_nodes"
    assert "latent" in latent_names, "latent (Latent) should be in latent_nodes"
    assert "y_obs" not in latent_names, "y_obs (Observable) should NOT be in latent_nodes"

    print("✓ test_latent_nodes_includes_priors passed")


def test_non_priors_excludes_priors():
    """Verify that get_non_root_nodes excludes root nodes (Priors)."""
    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")

    def latent_prior(mu_val, sigma_val):
        return tfd.Normal(mu_val, jp.abs(sigma_val))

    latent = paz.Latent(latent_prior, name="latent")(mu, sigma)

    def likelihood(latent_val):
        return tfd.Normal(latent_val, 0.1)

    y_obs = paz.Observable(likelihood, 0.0, name="y_obs")(latent)

    nodes = search_nodes([y_obs])
    edges = get_edges(nodes)
    dag = DAG([n.name for n in nodes], edges, "test")
    sorted_names = dag.sort_topologically()
    root_nodes = dag.root_nodes()

    non_priors = get_non_root_nodes(nodes, sorted_names, root_nodes)
    non_prior_names = [node.name for node in non_priors]

    print(f"Root nodes: {root_nodes}")
    print(f"Non-root nodes: {non_prior_names}")

    assert "mu" not in non_prior_names, "mu (root Prior) should NOT be in non_priors"
    assert "sigma" not in non_prior_names, "sigma (root Prior) should NOT be in non_priors"
    assert "latent" in non_prior_names, "latent (Latent) should be in non_priors"
    assert "y_obs" in non_prior_names, "y_obs (Observable) should be in non_priors"

    print("✓ test_non_priors_excludes_priors passed")


def test_sample_inverse_no_double_sampling():
    """Verify that _sample_inverse doesn't sample priors twice."""
    num_groups = 3

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), jp.abs(sigma)),
            reinterpreted_batch_ndims=1
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, jp.zeros(num_groups), name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "test")

    key = jax.random.PRNGKey(0)
    inverse_sample = model.sample_inverse(key, num_samples=1)

    # Check that we only have latent variables (mu, sigma, groups), not observables
    assert hasattr(inverse_sample, "mu")
    assert hasattr(inverse_sample, "sigma")
    assert hasattr(inverse_sample, "groups")
    assert not hasattr(inverse_sample, "y_obs"), "y_obs is Observable, not latent"

    # Check shapes
    assert inverse_sample.mu.shape == ()
    assert inverse_sample.sigma.shape == ()
    assert inverse_sample.groups.shape == (num_groups,)

    print("✓ test_sample_inverse_no_double_sampling passed")


def test_sample_forward_all_nodes():
    """Verify that sample_forward samples all nodes (priors, latents, observables)."""
    num_groups = 3

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), jp.abs(sigma)),
            reinterpreted_batch_ndims=1
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, jp.zeros(num_groups), name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "test")

    key = jax.random.PRNGKey(0)
    forward_sample = model.sample(key, num_samples=1)

    # Check that we have all nodes
    assert hasattr(forward_sample, "mu")
    assert hasattr(forward_sample, "sigma")
    assert hasattr(forward_sample, "groups")
    assert hasattr(forward_sample, "y_obs")

    # Check shapes
    assert forward_sample.mu.shape == ()
    assert forward_sample.sigma.shape == ()
    assert forward_sample.groups.shape == (num_groups,)
    assert forward_sample.y_obs.shape == (num_groups,)

    print("✓ test_sample_forward_all_nodes passed")


def test_filtering_consistency_batched():
    """Verify filtering works correctly with batched sampling."""
    num_groups = 3
    num_samples = 10

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), jp.abs(sigma)),
            reinterpreted_batch_ndims=1
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, jp.zeros(num_groups), name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "test")

    key = jax.random.PRNGKey(0)

    # Test inverse sampling
    inverse_samples = model.sample_inverse(key, num_samples=num_samples)
    assert inverse_samples.mu.shape == (num_samples,)
    assert inverse_samples.sigma.shape == (num_samples,)
    assert inverse_samples.groups.shape == (num_samples, num_groups)

    # Test forward sampling
    key, subkey = jax.random.split(key)
    forward_samples = model.sample(subkey, num_samples=num_samples)
    assert forward_samples.mu.shape == (num_samples,)
    assert forward_samples.sigma.shape == (num_samples,)
    assert forward_samples.groups.shape == (num_samples, num_groups)
    assert forward_samples.y_obs.shape == (num_samples, num_groups)

    print("✓ test_filtering_consistency_batched passed")


def test_hierarchical_regression_filtering():
    """Test filtering with hierarchical regression (multiple latents, multiple priors)."""
    num_groups = 5

    def slope_prior(mu_slope, sigma_slope):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_slope), jp.abs(sigma_slope)),
            reinterpreted_batch_ndims=1
        )

    def intercept_prior(mu_intercept, sigma_intercept):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_intercept), jp.abs(sigma_intercept)),
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

    def likelihood(slopes_val, intercepts_val):
        means = slopes_val[group_idx] * X + intercepts_val[group_idx]
        return tfd.Normal(means, 0.1)

    y_obs = paz.Observable(likelihood, jp.zeros(50), name="y_obs")(
        slopes, intercepts
    )

    priors = [mu_slope, mu_intercept, sigma_slope, sigma_intercept]
    model = paz.PGM(priors, [y_obs], "hierarchical")

    # Get internal node lists
    nodes = search_nodes([y_obs])
    latent_nodes = get_latent_nodes(nodes)
    latent_names = [n.name for n in latent_nodes]

    edges = get_edges(nodes)
    dag = DAG([n.name for n in nodes], edges, "test")
    sorted_names = dag.sort_topologically()
    root_nodes = dag.root_nodes()
    non_priors = get_non_root_nodes(nodes, sorted_names, root_nodes)
    non_prior_names = [n.name for n in non_priors]

    print(f"Latent nodes: {latent_names}")
    print(f"Non-prior nodes: {non_prior_names}")

    # Verify latent_nodes includes priors
    assert all(name in latent_names for name in ["mu_slope", "mu_intercept", "sigma_slope", "sigma_intercept"])
    assert "slopes" in latent_names
    assert "intercepts" in latent_names
    assert "y_obs" not in latent_names

    # Verify non_priors excludes priors
    assert all(name not in non_prior_names for name in ["mu_slope", "mu_intercept", "sigma_slope", "sigma_intercept"])
    assert "slopes" in non_prior_names
    assert "intercepts" in non_prior_names
    assert "y_obs" in non_prior_names

    # Test sampling
    key = jax.random.PRNGKey(42)
    inverse_samples = model.sample_inverse(key, num_samples=1)

    # Inverse samples should have all latent nodes
    assert hasattr(inverse_samples, "mu_slope")
    assert hasattr(inverse_samples, "mu_intercept")
    assert hasattr(inverse_samples, "sigma_slope")
    assert hasattr(inverse_samples, "sigma_intercept")
    assert hasattr(inverse_samples, "slopes")
    assert hasattr(inverse_samples, "intercepts")
    assert not hasattr(inverse_samples, "y_obs")

    key, subkey = jax.random.split(key)
    forward_samples = model.sample(subkey, num_samples=1)

    # Forward samples should have all nodes
    assert hasattr(forward_samples, "mu_slope")
    assert hasattr(forward_samples, "slopes")
    assert hasattr(forward_samples, "y_obs")

    print("✓ test_hierarchical_regression_filtering passed")


if __name__ == "__main__":
    print("Running node filtering tests...\n")
    test_latent_nodes_includes_priors()
    test_non_priors_excludes_priors()
    test_sample_inverse_no_double_sampling()
    test_sample_forward_all_nodes()
    test_filtering_consistency_batched()
    test_hierarchical_regression_filtering()
    print("\n✓ All node filtering tests passed!")
