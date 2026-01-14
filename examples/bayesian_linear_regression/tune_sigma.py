import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz

tfd = tfp.distributions
tfb = tfp.bijectors


def build_model(inputs, observations, low, high):
    def likelihood(inputs):
        def apply(mean, bias, stdv):
            return tfd.Normal(mean * inputs + bias, stdv)

        return apply

    mean = paz.Prior("mean", tfd.Normal(0.0, 1.0))
    bias = paz.Prior("bias", tfd.Normal(0.0, 1.0))
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    stdv = paz.Prior("stdv", tfd.Uniform(low, high), bijector=bijector)
    y_pred = paz.Observable("y_pred", likelihood(inputs), observations)(
        mean, bias, stdv
    )
    return paz.PGM([mean, bias, stdv], [y_pred], "line"), bijector


inputs = jp.linspace(0.0, 1.0, 200)
observations = 0.5 * inputs + 0.1 + 0.05 * jp.sin(50.0 * inputs)
low, high = 0.001, 0.3
model, bijector = build_model(inputs, observations, low, high)

log_density_fn = lambda params: model.apply(params).log_prob_sum

num_chains = 4
num_tune_steps = 1000
num_episodes = 50
sigma_0 = 0.05
num_samples = 5000
burn_in = 1000

key = jax.random.PRNGKey(0)
key, init_key, tune_key, sample_key = jax.random.split(key, 4)
positions = model.sample_inverse(init_key, num_chains)

tune = paz.Tuner(log_density_fn, positions, num_chains)
tuned_sigma, tune_infos = tune(tune_key, num_tune_steps, num_episodes, sigma_0)

episode_acceptance = tune_infos.acceptance_rate.mean(axis=1)
print("Tuning results")
print(f"  Final episode acceptance rate: {episode_acceptance[-1]:.3f}")
print(f"  Tuned sigma (mean across chains): {tuned_sigma:.5f}")

samples, infos = paz.metropolis_hastings.sample(
    sample_key,
    log_density_fn,
    positions,
    tuned_sigma,
    num_samples,
    num_chains,
)

acceptance_rate = infos.acceptance_rate[burn_in:].mean()
posterior = jax.tree.map(lambda x: x[burn_in:], samples)

print("Posterior estimates")
print(f"  Mean: {posterior.position.mean.mean():.4f}")
print(f"  Bias: {posterior.position.bias.mean():.4f}")
print(f"  Stdv: {bijector(posterior.position.stdv).mean():.4f}")
print(f"  Acceptance rate: {acceptance_rate:.3f}")
