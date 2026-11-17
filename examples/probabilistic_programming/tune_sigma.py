# Compare a posterior sampled with a tuned proposal sigma against one sampled
# with the initial sigma.

import argparse
from collections import namedtuple

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot

TrueParameters = namedtuple("TrueParameters", ["mean", "bias", "stdv"])


def build_model(inputs, low, high):
    standard = paz.distributions.Normal(0.0, 1.0)
    mean = paz.Prior(standard, name="mean")
    bias = paz.Prior(standard, name="bias")
    affine = [paz.bijectors.Shift(low), paz.bijectors.Scale(high - low)]
    bijector = paz.bijectors.Chain(affine + [paz.bijectors.Sigmoid()])
    uniform = paz.distributions.Uniform(low, high)
    stdv = paz.Prior(uniform, bijector, "stdv")
    y = paz.Observable(Likelihood(inputs), name="y_pred")(mean, bias, stdv)
    return paz.PGM([mean, bias, stdv], [y], "line")


def Likelihood(inputs):
    def apply(mean, bias, stdv):
        return paz.distributions.Normal(mean * inputs + bias, stdv)

    return apply


def print_estimates(posterior, truth):
    for name in truth._fields:
        estimate = getattr(posterior.samples, name).mean()
        print(f"  {name}: {estimate:.4f} (true: {getattr(truth, name):.3f})")
    rate = posterior.infos.acceptance_rate.mean()
    print(f"  acceptance rate: {rate:.3f}")


def plot_posterior(inputs, observations, posterior, truth, density_draw, name):
    samples = posterior.samples
    rate = posterior.infos.acceptance_rate.mean()
    figure, axes = plot.subplots(nrows=2, ncols=2, figsize=(12, 8))
    figure.suptitle(f"Posterior with {name} (acceptance {rate:.3f})")
    plot_line_traces(axes[0, 0], samples, truth)
    plot_stdv_trace(axes[0, 1], samples.stdv, truth.stdv)
    plot_joint(axes[1, 0], samples, truth)
    plot_predictive(axes[1, 1], inputs, samples)
    plot_references(axes[1, 1], inputs, observations, truth, density_draw)
    figure.tight_layout()


def plot_line_traces(axis, samples, truth):
    plot.line(*average_chains(samples.mean), axis, "C0", label="mean")
    plot.line(*average_chains(samples.bias), axis, "C1", label="bias")
    plot.hline(truth.mean, axis, color="C0", label="true mean")
    plot.hline(truth.bias, axis, color="C1", label="true bias")
    plot.set_labels(axis, x="Sample", y="Value")
    plot.legend(axis, loc="upper right")
    plot.clean(axis)
    axis.set_title("Chain-averaged traces")


def average_chains(samples):
    return jp.arange(len(samples)) + 1, samples.mean(axis=1)


def plot_stdv_trace(axis, stdv_samples, true_stdv):
    plot.line(*average_chains(stdv_samples), axis, "C2")
    plot.hline(true_stdv, axis, color="C2", label="true stdv")
    plot.set_labels(axis, x="Sample", y="Stdv")
    plot.legend(axis, loc="upper right")
    plot.clean(axis)
    axis.set_title("Noise scale trace")


def plot_joint(axis, samples, truth):
    means, biases = samples.mean.flatten(), samples.bias.flatten()
    plot.scatter(means, biases, axis, alpha=0.2, s=10, color="C0")
    kwargs = {"color": "black", "s": 80, "marker": "*", "label": "true"}
    axis.scatter(truth.mean, truth.bias, **kwargs)
    plot.set_labels(axis, x="mean", y="bias")
    plot.legend(axis, loc="upper right")
    plot.clean(axis)
    axis.set_title("Posterior mean vs bias")


def plot_predictive(axis, inputs, samples, num_draws=150):
    means, biases = samples.mean.reshape(-1, 1), samples.bias.reshape(-1, 1)
    predictions = means * inputs + biases
    low, high = jp.percentile(predictions, jp.array([5.0, 95.0]), axis=0)
    step = max(1, len(predictions) // num_draws)
    for draw in paz.to_numpy(predictions[::step]):
        plot.line(inputs, draw, axis, color="C0", alpha=0.05)
    plot.fill_between(inputs, low, high, axis, "C0", label="90% band")
    mean = predictions.mean(axis=0)
    plot.line(inputs, mean, axis, color="C0", label="posterior mean")


def plot_references(axis, inputs, observations, truth, density_draw):
    true_line = truth.mean * inputs + truth.bias
    plot.line(inputs, true_line, axis, "black", 1.5, "--", "true mean line")
    kwargs = {"color": "C3", "s": 16, "alpha": 0.8, "label": "observations"}
    plot.scatter(inputs, observations, axis, **kwargs)
    density_line = density_draw.mean * inputs + density_draw.bias
    kwargs = {"color": "tab:purple", "linestyle": "--"}
    plot.line(inputs, density_line, axis, label="gaussian approx", **kwargs)
    plot.set_labels(axis, x="inputs", y="observations")
    plot.legend(axis, loc="upper right")
    plot.clean(axis)
    axis.set_title("Posterior predictive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the proposal sigma")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--num_chains", type=int, default=4)
    parser.add_argument("--warmup", type=float, default=0.02)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--tune_steps", type=int, default=5000)
    parser.add_argument("--tune_episodes", type=int, default=50)
    args = parser.parse_args()

    # The residual is a sine of amplitude 0.05, so its stdv is 0.05 / sqrt(2).
    truth = TrueParameters(mean=0.5, bias=0.1, stdv=0.05 / 2**0.5)
    inputs = jp.linspace(0.0, 1.0, 200)
    wave = 0.05 * jp.sin(50.0 * inputs)
    observations = truth.mean * inputs + truth.bias + wave
    model = build_model(inputs, 0.001, 0.3)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 5)
    tuner_args = (args.sigma, args.tune_steps, args.tune_episodes)
    tuner = paz.AdaptiveStepTuner(*tuner_args)
    kwargs = {"sigma": args.sigma, "warmup": args.warmup, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    kwargs = {"num_samples": args.num_samples}
    tuned = model.infer(keys[1], observations, **kwargs)
    initial = model.infer(keys[2], observations, tuner=None, **kwargs)
    print(f"Tuned sigma: {tuned.config['sigma']:.5f}")

    plot.configure(fontsize=11)
    names = ("tuned sigma", "initial sigma")
    for name, posterior, key in zip(names, (tuned, initial), keys[3:]):
        print(f"Posterior estimates ({name})")
        print_estimates(posterior, truth)
        density = posterior.as_density(method="gaussian")
        density_draw = density.sample(key, num_samples=1)
        plot_args = (inputs, observations, posterior, truth)
        plot_posterior(*plot_args, density_draw, name)
    plot.show()
