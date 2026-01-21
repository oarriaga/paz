import paz
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def Likelihood(X):
    def apply(mean, bias, stdv):
        return tfd.Normal(mean * X + bias, stdv)

    return apply


X = jp.linspace(0, 1, 200)
data = 0.5 * X + 0.1 + 0.05 * jp.sin(50 * X)

mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
bias = paz.Prior(tfd.Normal(0.0, 1.0), name="bias")
low, high = 0.001, 0.3
bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
stdv = paz.Prior(tfd.Uniform(low, high), name="stdv", bijector=bijector)
y = paz.Observable(Likelihood(X), name="y_pred")(mean, bias, stdv)
model = paz.PGM([mean, bias, stdv], [y], "line")

keys = jax.random.split(jax.random.PRNGKey(888), 4)
samples = model.sample_inverse(keys[0], num_samples=100)

for mean, bias in zip(samples.mean, samples.bias):
    plt.plot(X, mean * X + bias, color="blue", alpha=0.2)
plt.plot(X, data, color="red", alpha=0.8)
plt.show()

num_chains = 4
num_samples = 10_000
burn_in = 0.1
sigma = 0.01

tuner = paz.AdaptiveStepTuner(sigma)
model.configure(num_chains=num_chains, warmup=burn_in, tuner=tuner)
posterior = model.infer(keys[1], data, num_samples=num_samples)
inverse_samples = posterior.inverse_samples
posterior_forward = posterior.samples

print(f"Mean acceptance rate: {posterior.infos.acceptance_rate.mean():.3f}")
print(f"Posterior mean: {inverse_samples.position.mean.mean():.4f} (true: 0.5)")
print(f"Posterior bias: {inverse_samples.position.bias.mean():.4f} (true: 0.1)")
print(f"Posterior stdv: {posterior_forward.stdv.mean():.4f} " f"(true: ~0.05)")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for chain in range(num_chains):
    mean = inverse_samples.position.mean[:, chain]
    bias = inverse_samples.position.bias[:, chain]
    plt.scatter(mean, bias, alpha=0.1, s=1)
plt.xlabel("mean")
plt.ylabel("bias")
plt.title("Posterior samples (unconstrained)")

plt.subplot(1, 2, 2)
posterior_density = posterior.as_density(method="gaussian")
samples = posterior_density.sample(keys[3], int(num_samples * 0.2))
for mean, bias in zip(samples.mean, samples.bias):
    plt.plot(X, mean * X + bias, color="blue", alpha=0.1)
plt.plot(X, data, color="red", alpha=0.8, label="data")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Posterior predictive")
plt.legend()
plt.show()
