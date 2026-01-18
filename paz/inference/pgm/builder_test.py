import jax.numpy as jp
import paz
from paz.abstract.dag import DAG
from paz.inference.pgm import (
    get_edges,
    get_latent_nodes,
    get_non_root_nodes,
    search_nodes,
)
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def test_latent_nodes_includes_priors():
    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")

    def latent_prior(mu_val, sigma_val):
        return tfd.Normal(mu_val, jp.abs(sigma_val))

    latent = paz.Latent(latent_prior, name="latent")(mu, sigma)

    def likelihood(latent_val):
        return tfd.Normal(latent_val, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(latent)
    nodes = search_nodes([y_obs])
    latent_nodes = get_latent_nodes(nodes)
    latent_names = [node.name for node in latent_nodes]

    assert "mu" in latent_names
    assert "sigma" in latent_names
    assert "latent" in latent_names
    assert "y_obs" not in latent_names


def test_non_priors_excludes_priors():
    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")

    def latent_prior(mu_val, sigma_val):
        return tfd.Normal(mu_val, jp.abs(sigma_val))

    latent = paz.Latent(latent_prior, name="latent")(mu, sigma)

    def likelihood(latent_val):
        return tfd.Normal(latent_val, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(latent)

    nodes = search_nodes([y_obs])
    edges = get_edges(nodes)
    dag = DAG([n.name for n in nodes], edges, "test")
    sorted_names = dag.sort_topologically()
    root_nodes = dag.root_nodes()

    non_priors = get_non_root_nodes(nodes, sorted_names, root_nodes)
    non_prior_names = [node.name for node in non_priors]

    assert "mu" not in non_prior_names
    assert "sigma" not in non_prior_names
    assert "latent" in non_prior_names
    assert "y_obs" in non_prior_names
