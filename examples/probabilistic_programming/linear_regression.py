import argparse

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot


def build_model(X, low, high):
    standard = paz.distributions.Normal(0.0, 1.0)
    mean = paz.Prior(standard, name="mean")
    bias = paz.Prior(standard, name="bias")
    affine = [paz.bijectors.Shift(low), paz.bijectors.Scale(high - low)]
    bijector = paz.bijectors.Chain(affine + [paz.bijectors.Sigmoid()])
    uniform = paz.distributions.Uniform(low, high)
    stdv = paz.Prior(uniform, bijector, "stdv")
    y = paz.Observable(Likelihood(X), name="y_pred")(mean, bias, stdv)
    return paz.PGM([mean, bias, stdv], [y], "line")


def Likelihood(X):
    def apply(mean, bias, stdv):
        return paz.distributions.Normal(mean * X + bias, stdv)

    return apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian linear regression")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--num_samples", type=int, default=20_000)
    parser.add_argument("--num_chains", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=0.15)
    parser.add_argument("--sigma", type=float, default=0.01)
    args = parser.parse_args()

    true_mean, true_bias = 0.5, 0.1
    X = jp.linspace(0, 1, 200)
    data = true_mean * X + true_bias + 0.05 * jp.sin(50 * X)
    model = build_model(X, 0.001, 0.3)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 4)
    plot.configure(fontsize=14)

    prior_samples = model.sample(keys[0], num_samples=100)
    figure, axis = plot.subplots()
    kwargs = {"color": plot.BLUE_GREY.primary, "alpha": 0.2, "num_lines": 50}
    plot.prior_predictive(X, prior_samples.y_pred, axis, **kwargs)
    plot.scatter(X, data, axis, s=2, color=plot.EARTH.primary, alpha=0.8)
    plot.set_labels(axis, x="X", y="y")
    plot.clean(axis)
    axis.set_title("Prior predictive")
    plot.show()

    tuner = paz.AdaptiveStepTuner(args.sigma)
    kwargs = {"warmup": args.warmup, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    posterior = model.infer(keys[1], data, num_samples=args.num_samples)
    samples = posterior.samples
    acceptance_rates = posterior.infos.acceptance_rate
    print(f"Mean acceptance rate: {acceptance_rates.mean():.3f}")
    print(f"Posterior mean: {samples.mean.mean():.4f} (true: {true_mean})")
    print(f"Posterior bias: {samples.bias.mean():.4f} (true: {true_bias})")
    # The residual is a sine of amplitude 0.05, so its stdv is 0.05 / sqrt(2).
    print(f"Posterior stdv: {samples.stdv.mean():.4f} (true: 0.0354)")

    traces = {"mean": samples.mean, "bias": samples.bias}
    plot.trace_panel(traces, title="MCMC Traces")
    plot.show()

    figure, axis = plot.subplots()
    for chain in range(args.num_chains):
        chain_mean = samples.mean[:, chain]
        chain_bias = samples.bias[:, chain]
        plot.scatter(chain_mean, chain_bias, axis, s=1, alpha=0.1)
    plot.set_labels(axis, x="mean", y="bias")
    plot.clean(axis)
    axis.set_title("Joint posterior (constrained space)")
    plot.show()

    means, biases = samples.mean.flatten(), samples.bias.flatten()
    figure, axis = plot.subplots()
    plot.contour(means, biases, axis, levels=15)
    plot.vline(true_mean, axis, color=plot.EARTH.primary, label="true mean")
    plot.hline(true_bias, axis, color=plot.EARTH.secondary, label="true bias")
    plot.set_labels(axis, x="mean", y="bias")
    plot.clean(axis)
    axis.legend()
    axis.set_title("Joint posterior contours")
    plot.show()

    density = posterior.as_density(method="gaussian")
    predictions = density.sample(keys[3], 50)
    figure, axis = plot.subplots()
    kwargs = {"color": plot.BLUE_GREY.primary, "num_lines": 100, "alpha": 0.1}
    plot.posterior_predictive(X, predictions.y_pred, axis=axis, **kwargs)
    plot.scatter(X, data, axis, s=8, color=plot.EARTH.primary, alpha=0.8)
    plot.set_labels(axis, x="X", y="y")
    plot.clean(axis)
    axis.set_title("Posterior predictive")
    plot.show()

    figure, axis = plot.subplots()
    plot.diagnostics(acceptance_rates, axis)
    axis.set_title("Acceptance rates per chain")
    plot.show()

    true_values = {"mean": true_mean, "bias": true_bias}
    plot.corner({"mean": means, "bias": biases}, true_values)
    plot.show()
