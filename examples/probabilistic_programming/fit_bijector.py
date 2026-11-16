"""
Fit a bijector so a base distribution matches a target distribution.
"""

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot


def build_transformed(source, bijector):
    return paz.distributions.TransformedDistribution(
        distribution=source,
        bijector=bijector,
    )


key = jax.random.PRNGKey(7)
source = paz.distributions.Normal(0.0, 1.0)
target = paz.distributions.LogNormal(0.4, 0.6)
affine = [paz.bijectors.Shift(0.0), paz.bijectors.Scale(1.0)]
initial_bijector = paz.bijectors.Chain(affine)
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
after_dist = build_transformed(source, fitted_prior.bijector)

plot.configure(fontsize=12)
figure, axes = plot.subplots(ncols=2, figsize=(12, 4))

plot.line(x, target.prob(x), axes[0], label="Target", linewidth=2)
plot.line(x, before_dist.prob(x), axes[0], linestyle="--", label="Before fit")
plot.line(x, after_dist.prob(x), axes[0], linestyle="-.", label="After fit")
plot.set_labels(axes[0], x="x", y="Density")
plot.legend(axes[0])
plot.clean(axes[0])
axes[0].set_title("Transformed distribution")

plot.line(jp.arange(len(losses)), losses, axes[1], color="black")
plot.set_labels(axes[1], x="Step", y="NLL")
plot.clean(axes[1])
axes[1].set_title("Fit loss")

plot.show()
