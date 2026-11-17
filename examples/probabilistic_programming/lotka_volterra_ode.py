import argparse

import jax
import jax.numpy as jp
import numpy as np
from jax.experimental.ode import odeint
from keras.utils import get_file

import paz
import paz.utils.plot as plot
from paz import distributions as pzd


def load_lynx_hare():
    url = "https://raw.githubusercontent.com/pyro-ppl/datasets/master/"
    path = get_file("LynxHare.txt", url + "LynxHare.txt", cache_subdir="paz")
    table = np.loadtxt(path)
    return table[:, 0], jp.array(table[:, 1:])


def Likelihood(times):
    def apply(z_init, theta, sigma):
        kwargs = {"rtol": 1e-6, "atol": 1e-5, "mxstep": 1000}
        z = odeint(compute_derivative, z_init, times, theta, **kwargs)
        return pzd.LogNormal(jp.log(z), sigma)

    return apply


def compute_derivative(z, time, theta):
    hare, lynx = z
    alpha, beta, gamma, delta = theta
    hare_rate = (alpha - beta * lynx) * hare
    lynx_rate = (-gamma + delta * hare) * lynx
    return jp.stack([hare_rate, lynx_rate])


def compute_initial_values(data, period, sigma):
    # Chains started from prior draws stay in modes with the wrong period.
    rate = 2 * jp.pi / period
    hare_mean, lynx_mean = data.mean(axis=0)
    theta = jp.array([rate, rate / lynx_mean, rate, rate / hare_mean])
    return {"z_init": data[0], "theta": theta, "sigma": jp.full(2, sigma)}


def plot_population(axis, years, data, predictions, color, name, marker, line):
    low, high = jp.percentile(predictions, jp.array([10, 90]), axis=0)
    mean = predictions.mean(axis=0)
    plot.line(years, mean, axis, color, 1.5, line, f"pred {name}")
    plot.fill_between(years, low, high, axis, color, alpha=0.25)
    kwargs = {"mfc": "none", "ms": 4, "label": f"true {name}"}
    axis.plot(years, data, color + marker, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predator-prey model")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--tune_steps", type=int, default=500)
    parser.add_argument("--tune_episodes", type=int, default=10)
    parser.add_argument("--num_chains", type=int, default=32)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--initial_period", type=float, default=10.0)
    parser.add_argument("--initial_sigma", type=float, default=0.4)
    parser.add_argument("--num_predictions", type=int, default=1000)
    parser.add_argument("--plot_path", default="lotka_volterra_ode.png")
    args = parser.parse_args()

    # 200 steps of 32 chains take 0.2 s on CPU and 4.4 s on a 4 GB GPU.
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    years, data = load_lynx_hare()
    times = jp.arange(float(len(years)))

    positive = paz.bijectors.Exp()
    z_init_prior = pzd.LogNormal(jp.full(2, jp.log(10.0)), 1.0)
    z_init = paz.Prior(z_init_prior, positive, "z_init")
    theta_loc = jp.array([1.0, 0.05, 1.0, 0.05])
    theta_scale = jp.array([0.5, 0.05, 0.5, 0.05])
    theta_prior = pzd.TruncatedNormal(theta_loc, theta_scale, 0.0, jp.inf)
    theta = paz.Prior(theta_prior, positive, "theta")
    sigma = paz.Prior(pzd.LogNormal(jp.full(2, -1.0), 1.0), positive, "sigma")
    y = paz.Observable(Likelihood(times), name="y")(z_init, theta, sigma)
    model = paz.PGM([z_init, theta, sigma], [y], "lotka_volterra")

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    init = compute_initial_values(data, args.initial_period, args.initial_sigma)
    shape = (args.num_chains,)
    init = jax.tree.map(lambda x: jp.broadcast_to(x, shape + x.shape), init)
    tuner_args = (args.sigma, args.tune_steps, args.tune_episodes)
    tuner = paz.AdaptiveCovarianceTuner(*tuner_args)
    kwargs = {"warmup": args.warmup, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    kwargs = {"num_samples": args.num_samples, "init": init, "space": "fwd"}
    posterior = model.infer(keys[0], data, **kwargs)
    rate = posterior.diagnostics()["mean_acceptance_rate"]
    print(f"Mean acceptance rate: {rate:.3f}")
    for name in ("z_init", "theta", "sigma"):
        samples = getattr(posterior.samples, name)
        mean, stdv = samples.mean(axis=(0, 1)), samples.std(axis=(0, 1))
        print(f"{name} mean: {mean} stdv: {stdv}")

    predictions = posterior.sample(keys[1], args.num_predictions).y
    figure, axis = plot.subplots(figsize=(8, 6))
    hare_args = (data[:, 0], predictions[..., 0], "k", "hare", "o", "-.")
    lynx_args = (data[:, 1], predictions[..., 1], "b", "lynx", "x", "--")
    plot_population(axis, years, *hare_args)
    plot_population(axis, years, *lynx_args)
    plot.set_labels(axis, x="year", y="population (in thousands)")
    plot.set_limits(axis, y=(0, 160))
    plot.legend(axis)
    axis.set_title("Posterior predictive (80% CI) with predator-prey pattern.")
    plot.show()
