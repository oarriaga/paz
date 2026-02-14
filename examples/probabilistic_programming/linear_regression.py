import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.plot as plot

tfd = tfp.distributions
tfb = tfp.bijectors


def Likelihood(X):
    def apply(mean, bias, stdv):
        return tfd.Normal(mean * X + bias, stdv)

    return apply


# Building data
X = jp.linspace(0, 1, 200)
data = 0.5 * X + 0.1 + 0.05 * jp.sin(50 * X)

# Model definition
mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
bias = paz.Prior(tfd.Normal(0.0, 1.0), name="bias")
low, high = 0.001, 0.3
bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
stdv = paz.Prior(tfd.Uniform(low, high), name="stdv", bijector=bijector)
y = paz.Observable(Likelihood(X), name="y_pred")(mean, bias, stdv)
model = paz.PGM([mean, bias, stdv], [y], "line")

# Prior predictive samples
keys = jax.random.split(jax.random.PRNGKey(888), 4)
prior_samples = model.sample(keys[0], num_samples=100)

# Configure plotting
plot.configure(fontsize=14, latex=False)
figure, axis = plot.subplots()
kwargs = {"color": plot.BLUE_GREY.primary, "alpha": 0.2, "num_lines": 50}
plot.prior_predictive(X, prior_samples.y_pred, axis, **kwargs)
plot.scatter(X, data, axis, s=2, color=plot.EARTH.primary, alpha=0.8)
plot.set_labels(axis, x="X", y="y")
plot.clean(axis)
axis.set_title("Prior predictive")
plot.show()

# Inference
num_chains, num_samples = 10, 20_000
tuner = paz.AdaptiveStepTuner(sigma=0.01)
model.configure(num_chains=num_chains, warmup=0.15, tuner=tuner)
posterior = model.infer(keys[1], data, num_samples=num_samples)
inverse_samples = posterior.inverse_samples
forward_samples = posterior.samples

# Print results
print(f"Mean acceptance rate: {posterior.infos.acceptance_rate.mean():.3f}")
print(f"Posterior mean: {forward_samples.mean.mean():.4f} (true: 0.5)")
print(f"Posterior bias: {forward_samples.bias.mean():.4f} (true: 0.1)")
print(f"Posterior stdv: {forward_samples.stdv.mean():.4f} (true: ~0.05)")

# Trace panel for convergence diagnostics
samples = {"mean": inverse_samples.mean, "bias": inverse_samples.bias}
plot.trace_panel(samples, title="MCMC Traces")
plot.show()

# Posterior scatter (joint posterior of mean vs bias)
figure, axis = plot.subplots()
for chain in range(num_chains):
    chain_mean = forward_samples.mean[:, chain]
    chain_bias = forward_samples.bias[:, chain]
    plot.scatter(chain_mean, chain_bias, axis, s=1, alpha=0.1)
plot.set_labels(axis, x="mean", y="bias")
plot.clean(axis)
axis.set_title("Joint posterior (constrained space)")
plot.show()

# Contour plot of joint posterior
all_means = forward_samples.mean.flatten()
all_biases = forward_samples.bias.flatten()
figure, axis = plot.subplots()
plot.contour(all_means, all_biases, axis, levels=15, cmap="viridis")
plot.vline(0.5, axis, color=plot.EARTH.primary, label="true mean")
plot.hline(0.1, axis, color=plot.EARTH.secondary, label="true bias")
plot.set_labels(axis, x="mean", y="bias")
plot.clean(axis)
axis.legend()
axis.set_title("Joint posterior contours")
plot.show()

# Posterior predictive
posterior_density = posterior.as_density(method="gaussian")
pred_samples = posterior_density.sample(keys[3], 50)
figure, axis = plot.subplots()
kwargs = {"color": plot.BLUE_GREY.primary, "num_lines": 100, "alpha": 0.1}
plot.posterior_predictive(X, pred_samples.y_pred, axis=axis, **kwargs)
plot.scatter(X, data, axis, s=8, color=plot.EARTH.primary, alpha=0.8)
plot.set_labels(axis, x="X", y="y")
plot.clean(axis)
axis.set_title("Posterior predictive")
plot.show()


# Diagnostics (acceptance rate per chain)
figure, axis = plot.subplots()
plot.diagnostics(posterior.infos.acceptance_rate, axis)
plot.clean(axis)
axis.set_title("Acceptance rates per chain")
plot.show()

# Corner plot showing all pairwise relationships
samples = {"mean": all_means, "bias": all_biases}
plot.corner(samples, true_values={"mean": 0.5, "bias": 0.1})
plot.show()
