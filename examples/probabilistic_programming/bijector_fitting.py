"""
Fit a bijector so a base distribution matches a target distribution.
"""

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import paz


tfd = tfp.distributions
tfb = tfp.bijectors


def build_transformed(source, bijector):
    return tfd.TransformedDistribution(
        distribution=source,
        bijector=bijector,
    )


def main():
    key = jax.random.PRNGKey(7)
    source = tfd.Normal(0.0, 1.0)
    target = tfd.LogNormal(0.4, 0.6)
    initial_bijector = tfb.Chain([tfb.Shift(0.0), tfb.Scale(1.0)])
    prior = paz.Prior(source, bijector=initial_bijector, name="z")

    fitted_prior, losses = prior.fit_bijector(
        key,
        target,
        num_samples=5000,
        num_steps=5_000,
        return_losses=True,
    )

    x = jp.linspace(0.001, 6.0, 300)
    before_dist = build_transformed(source, initial_bijector)
    after_dist = build_transformed(source, fitted_prior.metadata.bijector)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, target.prob(x), label="Target", linewidth=2)
    axes[0].plot(x, before_dist.prob(x), "--", label="Before fit")
    axes[0].plot(x, after_dist.prob(x), "-.", label="After fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Transformed distribution")
    axes[0].legend()

    axes[1].plot(losses, color="black")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("NLL")
    axes[1].set_title("Fit loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
