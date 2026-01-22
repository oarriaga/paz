import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz

tfd = tfp.distributions
tfb = tfp.bijectors


def Likelihood(X):
    def apply(mean, bias, stdv):
        return tfd.Normal(mean * X + bias, stdv)

    return apply


# Data
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

# Prior predictive
keys = jax.random.split(jax.random.PRNGKey(888), 4)
prior_samples = model.sample(keys[0], num_samples=100)

# Configure plotting
paz.plot.configure(fontsize=14, latex=False)
figure, axis = paz.plot.subplots()
color = paz.plot.BLUE_GREY.primary
paz.plot.prior_predictive(X, prior_samples.y_pred, axis, color=color)
paz.plot.scatter(X, data, axis, s=2, color=paz.plot.EARTH.primary, alpha=0.8)
paz.plot.set_labels(axis, x="X", y="y")
paz.plot.clean(axis)
axis.set_title("Prior predictive")
paz.plot.show()

# Inference
num_chains = 4
num_samples = 10_000
burn_in = 0.2
sigma = 0.01

tuner = paz.AdaptiveStepTuner(sigma)
model.configure(num_chains=num_chains, warmup=burn_in, tuner=tuner)
posterior = model.infer(keys[1], data, num_samples=num_samples)
inverse_samples = posterior.inverse_samples
posterior_forward = posterior.samples

# Print results
print(f"Mean acceptance rate: {posterior.infos.acceptance_rate.mean():.3f}")
print(f"Posterior mean: {inverse_samples.mean.mean():.4f} (true: 0.5)")
print(f"Posterior bias: {inverse_samples.bias.mean():.4f} (true: 0.1)")
print(f"Posterior stdv: {posterior_forward.stdv.mean():.4f} (true: ~0.05)")

# Trace panel for convergence diagnostics
trace = {"mean": inverse_samples.mean, "bias": inverse_samples.bias}
paz.plot.trace_panel(trace, title="MCMC Traces")
paz.plot.show()

# Posterior scatter (joint posterior of mean vs bias)
figure, axis = paz.plot.subplots()
for chain in range(num_chains):
    chain_mean = inverse_samples.mean[:, chain]
    chain_bias = inverse_samples.bias[:, chain]
    paz.plot.scatter(chain_mean, chain_bias, axis, s=1, alpha=0.1)
paz.plot.set_labels(axis, x="mean", y="bias")
paz.plot.clean(axis)
axis.set_title("Joint posterior (unconstrained space)")
paz.plot.show()

# Contour paz.plot of joint posterior
all_means = inverse_samples.mean.flatten()
all_biases = inverse_samples.bias.flatten()
figure, axis = paz.plot.subplots()
paz.plot.contour(all_means, all_biases, axis, levels=15, cmap="viridis")
color_1, color_2 = paz.plot.EARTH.primary, paz.plot.EARTH.secondary
paz.plot.vline(0.5, axis, color=color_1, linestyle="--", label="true mean")
paz.plot.hline(0.1, axis, color=color_2, linestyle="--", label="true bias")
paz.plot.set_labels(axis, x="mean", y="bias")
paz.plot.clean(axis)
axis.legend()
axis.set_title("Joint posterior contours")
paz.plot.show()

# Posterior predictive
posterior_density = posterior.as_density(method="gaussian")
pred_samples = posterior_density.sample(keys[3], int(num_samples * 0.2))
y_pred = pred_samples.y_pred

figure, axis = paz.plot.subplots()
color = paz.plot.BLUE_GREY.primary
paz.plot.posterior_predictive(X, y_pred, axis=axis, color=color, num_lines=100)
paz.plot.scatter(X, data, axis, s=8, color=paz.plot.EARTH.primary, alpha=0.8)
paz.plot.set_labels(axis, x="X", y="y")
paz.plot.clean(axis)
axis.set_title("Posterior predictive")
paz.plot.show()


# Diagnostics (acceptance rate per chain)
figure, axis = paz.plot.subplots()
paz.plot.diagnostics(posterior.infos.acceptance_rate, axis)
paz.plot.clean(axis)
axis.set_title("Acceptance rates per chain")
paz.plot.show()

# Corner paz.plot showing all pairwise relationships
paz.plot.corner(
    {"mean": all_means, "bias": all_biases},
    true_values={"mean": 0.5, "bias": 0.1},
)
paz.plot.show()
