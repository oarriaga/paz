import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

from paz.inference.discretizer import discretize, get_grid_values


tfd = tfp.distributions


def _plot_discretization(
    axis, distribution, min_val, max_val, num_steps, title
):
    xs = jp.linspace(min_val, max_val, 400)
    pdf = jp.exp(distribution.log_prob(xs))
    grid = get_grid_values(min_val, max_val, num_steps)
    categorical = discretize(distribution, min_val, max_val, num_steps)
    probs = categorical.probs_parameter()
    step_width = (max_val - min_val) / (num_steps - 1)
    discrete_density = probs / step_width
    prob_axis = axis.twinx()
    axis.plot(xs, pdf, color="black", linewidth=2, label="continuous pdf")
    axis.vlines(
        grid,
        0.0,
        discrete_density,
        colors="tab:blue",
        linewidth=3,
        alpha=0.8,
        label="discrete density",
        zorder=2,
    )
    prob_axis.scatter(
        grid,
        probs,
        color="tab:orange",
        s=18,
        alpha=0.7,
        label="discrete prob",
        zorder=3,
    )
    axis.set_title(title)
    axis.set_xlabel("value")
    axis.set_ylabel("density")
    prob_axis.set_ylabel("probability")
    handles, labels = axis.get_legend_handles_labels()
    prob_handles, prob_labels = prob_axis.get_legend_handles_labels()
    axis.legend(handles + prob_handles, labels + prob_labels, loc="upper right")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    normal = tfd.Normal(loc=0.0, scale=1.0)
    _plot_discretization(
        axes[0], normal, -3.0, 3.0, 31, "Normal discretization"
    )
    skewed = tfd.LogNormal(loc=0.0, scale=0.6)
    _plot_discretization(
        axes[1], skewed, 0.0, 6.0, 25, "LogNormal discretization"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
