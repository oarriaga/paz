import argparse

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot


def LogDensity(mean, stdv):
    def apply(position):
        normal = paz.distributions.Normal(mean, stdv)
        return normal.log_prob(position).sum()

    return apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metropolis-Hastings sampling")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--num_chains", type=int, default=5)
    args = parser.parse_args()

    mean = jp.array([1.5, 1.5])
    stdv = jp.array([0.5, 1.0])
    log_density = LogDensity(mean, stdv)
    key = jax.random.PRNGKey(args.seed)
    positions = jp.ones((args.num_chains, 2))
    shape = (args.num_samples, args.num_chains)
    sample_args = (key, log_density, positions, args.sigma, *shape)
    samples, infos = paz.metropolis_hastings.sample(*sample_args)
    print("Chain acceptance rate:", infos.is_accepted.mean(axis=0))
    print("Sample mean:", samples.position.mean(axis=(0, 1)), "true:", mean)
    print("Sample stdv:", samples.position.std(axis=(0, 1)), "true:", stdv)

    plot.configure(fontsize=12)
    figure, axis = plot.subplots()
    plot.trace_lines(samples.position[:, :, 1], axis)
    plot.set_labels(axis, x="sample", y="value")
    plot.clean(axis)
    plot.show()
