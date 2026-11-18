# Neal's funnel sampled in two equivalent forms.
#
# True model, 1 + num_dims dimensions:
#   v ~ Normal(0, 3)
#   x_i ~ Normal(0, exp(v / 2)) for i = 0, ..., num_dims - 1
# All x_i share the same v. Alone, v is exactly Normal(0, 3).
#
# Centered form: the sampler moves (v, x). From v = -10 to v = 10 the width
# of x grows by a factor of exp(10), so no single step size fits both ends.
#
# Non-centered form: the sampler moves (v, z) with z_i ~ Normal(0, 1), and
# x_i = exp(v / 2) * z_i is computed afterwards.
#
# Why both are the same distribution: if z ~ Normal(0, 1) then
# s * z ~ Normal(0, s). With s = exp(v / 2), x_i = exp(v / 2) * z_i is
# Normal(0, exp(v / 2)) for every v, which is the centered definition.
# In densities, the change of variables x = exp(v / 2) * z gives
#   p(v, z) = p(v, x) * exp(num_dims * v / 2)
# and the factor cancels the 1 / exp(v / 2) normalizer of each x_i:
#   log p(v, x) = -v^2 / 18 - num_dims * v / 2 - sum(x^2) / (2 exp(v))
#   log p(v, z) = -v^2 / 18 - sum(z^2) / 2
# up to constants. No term of log p(v, z) contains both v and z.
#
# This is not special to the Gaussian. It works whenever the child can be
# written as a fixed function of its parents and a noise variable whose
# distribution does not depend on them: any family defined by a shift and a
# stretch (Cauchy, Laplace, Logistic, Gumbel), a pure scale such as
# Exponential(rate) = Exponential(1) / rate, or a transform of those such
# as LogNormal. It does not carry over simply to parameters that change the
# shape (the shape of a Gamma or Beta) or to discrete variables.
#
# The figure shows two projections of the full distribution: the top row
# keeps only (x_0, v) and drops x_1, ..., x_{num_dims - 1}; the bottom row
# keeps only v. Every dropped x_i has the same funnel shape against v.

import argparse

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot


def build_centered_model(num_dims):
    v = paz.Prior(paz.distributions.Normal(0.0, 3.0), name="v")
    x = paz.Latent(Funnel(num_dims), name="x")(v)
    return paz.PGM([v], [x], "funnel_centered")


def Funnel(num_dims):
    def apply(v):
        return paz.distributions.Normal(jp.zeros(num_dims), jp.exp(v / 2))

    return apply


def build_noncentered_model(num_dims):
    v = paz.Prior(paz.distributions.Normal(0.0, 3.0), name="v")
    z = paz.Prior(paz.distributions.Normal(jp.zeros(num_dims), 1.0), name="z")
    return paz.PGM([v, z], [v, z], "funnel_noncentered")


def unpack_centered(samples):
    return samples.v, samples.x


def compute_noncentered(samples):
    return samples.v, jp.exp(samples.v / 2)[..., None] * samples.z


def print_summary(name, v, acceptance_rates, neck):
    chain_means = v.mean(axis=0)
    print(f"{name}: acceptance {acceptance_rates.mean():.3f}")
    print(f"  mean of v: {v.mean():.3f} (exact: 0)")
    print(f"  stdv of v: {v.std():.3f} (exact: 3)")
    # The neck is where the centered form fails: 15.87% of v is below -3.
    print(f"  fraction of v below {neck}: {(v < neck).mean():.4f}")
    print(f"  chain means: {chain_means.min():.3f} to {chain_means.max():.3f}")


def plot_funnel(axis, x_0, v, v_grid):
    plot.scatter(x_0, v, axis, color="C1", s=2, alpha=0.3)
    # 95% of the exact x_0 lies between these two curves at every v.
    width = 1.96 * jp.exp(v_grid / 2)
    plot.line(width, v_grid, axis, "black", 1.0, label="exact 95% width")
    plot.line(-width, v_grid, axis, "black", 1.0)
    plot.set_limits(axis, x=(-10, 10), y=(v_grid[0], v_grid[-1]))
    plot.set_labels(axis, x="x_0", y="v")
    plot.clean(axis)


def plot_marginal(axis, v, v_grid):
    bins = jp.linspace(v_grid[0], v_grid[-1], 61)
    plot.histogram(v, axis, bins, color="C1", label="sampler")
    exact = jax.scipy.stats.norm.pdf(v_grid, 0.0, 3.0)
    plot.line(v_grid, exact, axis, "black", 1.5, label="exact Normal(0, 3)")
    plot.set_limits(axis, x=(v_grid[0], v_grid[-1]))
    plot.set_labels(axis, x="v", y="Density")
    plot.clean(axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neal's funnel")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num_dims", type=int, default=9)
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--num_chains", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=5000)
    args = parser.parse_args()

    centered = (build_centered_model, unpack_centered)
    noncentered = (build_noncentered_model, compute_noncentered)
    forms = {"Centered": centered, "Non-centered": noncentered}
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    neck = -3.0
    exact_fraction = jax.scipy.stats.norm.cdf(neck, 0.0, 3.0)
    print(f"Exact fraction of v below {neck}: {exact_fraction:.4f}")

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(nrows=2, ncols=2, figsize=(11, 9))
    v_grid = jp.linspace(-10.0, 10.0, 200)
    for column, (name, (build_model, get_funnel)) in enumerate(forms.items()):
        model = build_model(args.num_dims)
        tuner = paz.AdaptiveStepTuner(args.sigma)
        kwargs = {"warmup": args.warmup, "sigma": args.sigma, "tuner": tuner}
        model.configure(num_chains=args.num_chains, **kwargs)
        posterior = model.infer(keys[0], {}, num_samples=args.num_samples)
        v, x = get_funnel(posterior.samples)
        print_summary(name, v, posterior.infos.acceptance_rate, neck)

        indices = jax.random.choice(keys[1], v.size, (args.num_points,))
        x_0 = x[..., 0].ravel()[indices]
        plot_funnel(axes[0, column], x_0, v.ravel()[indices], v_grid)
        plot_marginal(axes[1, column], v, v_grid)
        axes[0, column].set_title(name)
    plot.legend(axes[0, 0], loc="upper left")
    plot.legend(axes[1, 0], loc="upper left")
    figure.tight_layout()
    plot.show()
