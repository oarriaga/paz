import paz
import jax
import jax.numpy as jp
import paz.utils.plot as plot

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

true_mean = jp.array([1.5, 1.5])
true_stdv = jp.array([[0.5, 1.0], [1.0, 0.5]])
key = jax.random.PRNGKey(777)
sigma = 0.25
num_samples = 10_000
num_chains = 5
positions = jp.repeat(jp.expand_dims(jp.array([1.0, 1.0]), 0), num_chains, 0)


def log_density(position):
    return tfd.Normal(true_mean, true_stdv).log_prob(position).sum()


samples, states = paz.metropolis_hastings.sample(
    key, log_density, positions, sigma, num_samples, num_chains
)

plot.trace(samples.position[:, 0, 1], y_label="value")


def compute_acceptance_rate(infos):
    return jp.sum(infos.is_accepted, axis=0) / len(infos.is_accepted)


print("Chain acceptance rate:", compute_acceptance_rate(states))
