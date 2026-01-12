import paz
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def Likelihood(X):
    def apply(mean, bias, stdv):
        return tfd.Normal(jax.vmap(lambda x: mean * x + bias)(X), stdv)

    return apply


X = jp.linspace(0, 1, 200)
observations = 0.5 * X + 0.1 + 0.05 * jp.sin(50 * X)

mean = paz.Prior("mean", tfd.Normal(0.0, 1.0))
bias = paz.Prior("bias", tfd.Normal(0.0, 1.0))
low, high = 0.001, 0.3
bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
stdv = paz.Prior("stdv", tfd.Uniform(low, high), bijector=bijector)
y = paz.Observable("y_pred", Likelihood(X), observations)(mean, bias, stdv)
line = paz.PGM([mean, bias, stdv], [y], "line")

key = jax.random.PRNGKey(888)
for key in jax.random.split(key, num_samples := 100):
    sample = line.sample(key)
    plt.plot(X, sample.y_pred, color="blue", alpha=0.2)
plt.plot(X, observations, color="red", alpha=0.8)
plt.xlim(0, 1)
plt.ylim(-4, 4)
plt.show()

log_density_fn = lambda params: line.apply(params).log_prob_sum

num_chains = 4
num_samples = 5000
sigma = 0.01

key = jax.random.PRNGKey(888)
key, init_key = jax.random.split(key)
positions = line.sample_inverse(init_key, num_chains)

samples, infos = paz.metropolis_hastings.sample(
    key, log_density_fn, positions, sigma, num_samples, num_chains
)

burn_in = 1000
posterior = jax.tree.map(lambda x: x[burn_in:], samples)

print(f"Mean acceptance rate: {infos.acceptance_rate[burn_in:].mean():.3f}")
print(f"Posterior mean: {posterior.position.mean.mean():.4f} (true: 0.5)")
print(f"Posterior bias: {posterior.position.bias.mean():.4f} (true: 0.1)")
print(
    f"Posterior stdv: {bijector(posterior.position.stdv).mean():.4f} (true: ~0.05)"
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for chain in range(num_chains):
    plt.scatter(
        posterior.position.mean[:, chain],
        posterior.position.bias[:, chain],
        alpha=0.1,
        s=1,
    )
plt.xlabel("mean")
plt.ylabel("bias")
plt.title("Posterior samples (unconstrained)")

plt.subplot(1, 2, 2)
key, plot_key = jax.random.split(key)
for i, k in enumerate(jax.random.split(plot_key, int(num_samples * 0.2))):
    idx = jax.random.randint(k, (), 0, posterior.position.mean.size)
    m = posterior.position.mean.flatten()[idx]
    b = posterior.position.bias.flatten()[idx]
    plt.plot(X, m * X + b, color="blue", alpha=0.1)
plt.plot(X, observations, color="red", alpha=0.8, label="observations")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Posterior predictive")
plt.legend()
plt.tight_layout()
plt.show()
