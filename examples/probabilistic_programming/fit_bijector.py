# Fit a bijector so a base distribution matches a target distribution.
# Exp of a shifted and scaled Normal is a LogNormal, so the fit can be exact.

import argparse

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot
from paz import distributions as pzd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a bijector to a target")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_steps", type=int, default=5000)
    args = parser.parse_args()

    source = pzd.Normal(0.0, 1.0)
    target = pzd.LogNormal(0.4, 0.6)
    affine = [paz.bijectors.Shift(0.0), paz.bijectors.Scale(1.0)]
    initial_bijector = paz.bijectors.Chain([paz.bijectors.Exp()] + affine)
    prior = paz.Prior(source, initial_bijector, "z")

    key = jax.random.PRNGKey(args.seed)
    kwargs = {"num_samples": args.num_samples, "num_steps": args.num_steps}
    fitted = prior.fit_bijector(key, target, return_losses=True, **kwargs)
    fitted_prior, losses = fitted
    shift, scale = jax.tree.leaves(fitted_prior.bijector)
    print(f"Fitted shift: {shift:.3f} (target: 0.4)")
    print(f"Fitted scale: {scale:.3f} (target: 0.6)")

    x = jp.linspace(0.001, 6.0, 300)
    before = pzd.TransformedDistribution(source, initial_bijector)
    after = pzd.TransformedDistribution(source, fitted_prior.bijector)

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(ncols=2, figsize=(12, 4))
    plot.line(x, target.prob(x), axes[0], "black", label="Target")
    kwargs = {"linestyle": "--", "label": "Before fit"}
    plot.line(x, before.prob(x), axes[0], "C0", **kwargs)
    kwargs = {"linestyle": "-.", "label": "After fit"}
    plot.line(x, after.prob(x), axes[0], "C1", **kwargs)
    plot.set_labels(axes[0], x="x", y="Density")
    plot.legend(axes[0])
    plot.clean(axes[0])
    axes[0].set_title("Transformed distribution")

    plot.line(jp.arange(len(losses)), losses, axes[1], color="black")
    plot.set_labels(axes[1], x="Step", y="NLL")
    plot.clean(axes[1])
    axes[1].set_title("Fit loss")
    plot.show()
