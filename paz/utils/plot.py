import math
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

try:
    import arviz as az
except Exception:  # pragma: no cover - optional dependency
    az = None

try:
    import ternary
    from ternary.helpers import simplex_iterator
except Exception:  # pragma: no cover - optional dependency
    ternary = None
    simplex_iterator = None

# ---------------------------------------------------------------------------
# Color Palettes
# ---------------------------------------------------------------------------
Palette = namedtuple("Palette", ["primary", "secondary", "accent", "neutral"])

DANDELION = Palette(
    primary=(0.992, 0.737, 0.258),
    secondary="tab:blue",
    accent="tab:green",
    neutral=(0.662, 0.647, 0.576),
)

BLUE_GREY = Palette(
    primary="tab:blue",
    secondary=(0.662, 0.647, 0.576),
    accent="tab:purple",
    neutral="0.7",
)

GREEN_YELLOW = Palette(
    primary="#2ca02c",
    secondary="#d4b700",
    accent="tab:orange",
    neutral="0.6",
)

EARTH = Palette(
    primary="#d62728",
    secondary="#2ca02c",
    accent="#1f77b4",
    neutral="0.5",
)

DEFAULT_PALETTE = DANDELION


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
Config = namedtuple("Config", ["fontsize", "latex", "linewidth", "font"])


def configure(fontsize=16, latex=False, linewidth=2.0, font="serif"):
    """Set matplotlib defaults for publication-quality plots."""
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["lines.linewidth"] = linewidth
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    if latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.serif"] = ["Times New Roman", "Palatino"]
    return Config(fontsize, latex, linewidth, font)


# ---------------------------------------------------------------------------
# Axis utilities
# ---------------------------------------------------------------------------
def hide_spines(axis, which="default"):
    """Hide specified spines. Default hides top, right, and left."""
    if which == "default":
        which = ["top", "right", "left"]
    elif which == "all":
        which = ["top", "right", "left", "bottom"]
    elif which == "box":
        which = ["top", "right"]
    for spine in which:
        axis.spines[spine].set_visible(False)


def add_minor_ticks(axis):
    """Add minor tick locators to both axes."""
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())


def set_labels(axis, x=None, y=None, labelpad=10):
    """Set axis labels with optional padding."""
    if x is not None:
        axis.set_xlabel(x, labelpad=labelpad)
    if y is not None:
        axis.set_ylabel(y, labelpad=labelpad)


def set_limits(axis, x=None, y=None):
    """Set axis limits."""
    if x is not None:
        axis.set_xlim(x)
    if y is not None:
        axis.set_ylim(y)


def clean(axis, spines="default"):
    """Clean axis by hiding spines and adding minor ticks."""
    hide_spines(axis, spines)
    add_minor_ticks(axis)


def legend(axis, frameon=False, loc="best", fontsize=None):
    """Add legend with sensible defaults."""
    axis.legend(frameon=frameon, loc=loc, fontsize=fontsize)


# ---------------------------------------------------------------------------
# Core plot functions
# ---------------------------------------------------------------------------
def density(
    samples,
    axis=None,
    color=None,
    fill=True,
    alpha=0.3,
    label=None,
    linewidth=2.0,
):
    """KDE density plot for 1D samples."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    samples = np.asarray(samples).flatten()
    kde = stats.gaussian_kde(samples)
    x_grid = np.linspace(samples.min(), samples.max(), 200)
    y_values = kde(x_grid)
    axis.plot(x_grid, y_values, color=color, linewidth=linewidth, label=label)
    if fill:
        axis.fill_between(x_grid, y_values, alpha=alpha, color=color)
    return axis


def histogram(
    samples,
    axis=None,
    bins=50,
    density_norm=True,
    color=None,
    alpha=0.7,
    label=None,
    edgecolor="white",
):
    """Histogram with optional density normalization."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    samples = np.asarray(samples).flatten()
    axis.hist(
        samples,
        bins=bins,
        density=density_norm,
        alpha=alpha,
        color=color,
        label=label,
        edgecolor=edgecolor,
    )
    return axis


def trace_lines(samples, axis=None, alpha=0.7, linewidth=0.5, color=None):
    """Plot MCMC trace. Handles (num_samples,) or (num_samples, num_chains)."""
    if axis is None:
        _, axis = plt.subplots()
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    num_chains = samples.shape[1]
    if color is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(num_chains)]
    else:
        colors = [color] * num_chains
    for chain in range(num_chains):
        axis.plot(
            samples[:, chain],
            alpha=alpha,
            linewidth=linewidth,
            color=colors[chain],
        )
    return axis


def scatter(x, y, axis=None, alpha=0.5, s=10, color=None, label=None):
    """Scatter plot."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.scatter(x, y, alpha=alpha, s=s, color=color, label=label)
    return axis


def line(
    x,
    y,
    axis=None,
    color=None,
    linewidth=2.0,
    linestyle="-",
    label=None,
    alpha=1.0,
):
    """Simple line plot."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        alpha=alpha,
    )
    return axis


def fill_between(
    x, y_lower, y_upper, axis=None, color=None, alpha=0.2, label=None
):
    """Shaded region between two curves."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.fill_between(
        x, y_lower, y_upper, alpha=alpha, color=color, label=label
    )
    return axis


def line_with_band(
    x,
    y_mean,
    y_std=None,
    axis=None,
    color=None,
    alpha=0.2,
    label=None,
    linewidth=2.0,
):
    """Line plot with optional shaded uncertainty band."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.plot(x, y_mean, color=color, linewidth=linewidth, label=label)
    if y_std is not None:
        axis.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=color
        )
    return axis


def regression_lines(
    x, slopes, intercepts, axis=None, alpha=0.1, color=None, num_lines=50
):
    """Plot multiple regression lines from posterior samples."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    slopes = np.asarray(slopes).flatten()
    intercepts = np.asarray(intercepts).flatten()
    num_available = min(len(slopes), len(intercepts))
    num_to_plot = min(num_lines, num_available)
    indices = np.random.choice(num_available, num_to_plot, replace=False)
    for i in indices:
        y = slopes[i] * x + intercepts[i]
        axis.plot(x, y, alpha=alpha, color=color)
    return axis


def contour(
    samples_x, samples_y, axis=None, levels=10, cmap="viridis", fill=True
):
    """2D contour density plot."""
    if axis is None:
        _, axis = plt.subplots()
    samples_x = np.asarray(samples_x).flatten()
    samples_y = np.asarray(samples_y).flatten()
    x_grid = np.linspace(samples_x.min(), samples_x.max(), 100)
    y_grid = np.linspace(samples_y.min(), samples_y.max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([samples_x, samples_y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    if fill:
        axis.contourf(X, Y, Z, levels=levels, cmap=cmap)
    else:
        axis.contour(X, Y, Z, levels=levels, cmap=cmap)
    return axis


def vline(x, axis=None, color=None, linestyle="--", linewidth=1.5, label=None):
    """Add vertical line."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.secondary
    axis.axvline(
        x, color=color, linestyle=linestyle, linewidth=linewidth, label=label
    )
    return axis


def hline(y, axis=None, color=None, linestyle="--", linewidth=1.5, label=None):
    """Add horizontal line."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.secondary
    axis.axhline(
        y, color=color, linestyle=linestyle, linewidth=linewidth, label=label
    )
    return axis


def bar(x, heights, axis=None, color=None, alpha=0.8, width=0.8, label=None):
    """Bar plot."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.bar(x, heights, color=color, alpha=alpha, width=width, label=label)
    return axis


def imshow(
    data,
    axis=None,
    cmap="viridis",
    aspect="auto",
    origin="lower",
    extent=None,
    vmin=None,
    vmax=None,
    colorbar=True,
    colorbar_label=None,
):
    """Display 2D array as image/heatmap with optional colorbar."""
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    image = axis.imshow(
        data,
        cmap=cmap,
        aspect=aspect,
        origin=origin,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    if colorbar:
        cbar = fig.colorbar(image, ax=axis)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)
    return image


def stem(x, y, axis=None, color=None, linewidth=2, alpha=0.8, label=None):
    """Stem plot (lollipop chart) for discrete distributions."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    markerline, stemlines, baseline = axis.stem(x, y, basefmt=" ", label=label)
    plt.setp(stemlines, color=color, linewidth=linewidth, alpha=alpha)
    plt.setp(markerline, color=color, markersize=6)
    return axis


# ---------------------------------------------------------------------------
# Statistical comparison plots
# ---------------------------------------------------------------------------
def qq_plot(samples, mean, stdv, axis=None, color=None, line_color="red"):
    """Q-Q plot comparing samples to Normal(mean, stdv)."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    samples = np.asarray(samples).flatten()
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    theoretical_quantiles = stats.norm(mean, stdv).ppf(
        np.linspace(0.001, 0.999, n)
    )
    axis.scatter(
        theoretical_quantiles, sorted_samples, alpha=0.3, s=5, color=color
    )
    lims = [
        min(theoretical_quantiles.min(), sorted_samples.min()),
        max(theoretical_quantiles.max(), sorted_samples.max()),
    ]
    axis.plot(lims, lims, color=line_color, linestyle="--", linewidth=2)
    set_labels(axis, x="theoretical quantiles", y="sample quantiles")
    return axis


def compare_densities(
    x_range,
    densities,
    labels,
    axis=None,
    colors=None,
    linestyles=None,
    linewidth=2.0,
):
    """Plot multiple density curves on the same axis."""
    if axis is None:
        _, axis = plt.subplots()
    if colors is None:
        colors = [
            "tab:red",
            "tab:green",
            "tab:blue",
            "tab:purple",
            "tab:orange",
        ]
    if linestyles is None:
        linestyles = ["-", "--", ":", "-.", "-"]
    for i, (density_vals, label) in enumerate(zip(densities, labels)):
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        axis.plot(
            x_range,
            density_vals,
            color=color,
            linestyle=ls,
            linewidth=linewidth,
            label=label,
        )
    return axis


def discrete_posterior(
    support,
    probabilities,
    axis=None,
    true_value=None,
    color=None,
    true_color="tab:green",
):
    """Bar plot for discrete posterior probabilities."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.accent
    support = np.asarray(support)
    probabilities = np.asarray(probabilities)
    axis.bar(support, probabilities, color=color, alpha=0.7, width=0.8)
    if true_value is not None:
        vline(
            true_value,
            axis,
            color=true_color,
            linestyle="--",
            label="true value",
        )
    set_labels(axis, y="probability")
    set_limits(axis, y=(0, 1.05))
    return axis


def discretized_distribution(
    distribution, min_val, max_val, num_steps, axis=None, title=None
):
    """Plot continuous distribution alongside its discretized version.

    Shows three elements:
    - Continuous PDF as a line
    - Discrete density as stem plot (probability per unit width)
    - Discrete probabilities as scatter points on a twin y-axis
    """
    import jax.numpy as jp
    from paz.inference.discretizer import discretize, get_grid_values

    if axis is None:
        _, axis = plt.subplots()

    xs = jp.linspace(min_val, max_val, 400)
    pdf = jp.exp(distribution.log_prob(xs))
    grid = get_grid_values(min_val, max_val, num_steps)
    categorical = discretize(distribution, min_val, max_val, num_steps)
    probs = categorical.probs_parameter()
    step_width = (max_val - min_val) / (num_steps - 1)
    discrete_density = probs / step_width

    prob_axis = axis.twinx()
    line(xs, pdf, axis, color="black", linewidth=2, label="continuous pdf")
    stem(
        grid,
        discrete_density,
        axis,
        color="tab:blue",
        linewidth=3,
        alpha=0.8,
        label="discrete density",
    )
    scatter(
        grid,
        probs,
        prob_axis,
        color="tab:orange",
        s=18,
        alpha=0.7,
        label="discrete prob",
    )

    if title is not None:
        axis.set_title(title)
    set_labels(axis, x="value", y="density")
    prob_axis.set_ylabel("probability")

    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = prob_axis.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    return axis, prob_axis


# ---------------------------------------------------------------------------
# GMM plotting functions
# ---------------------------------------------------------------------------
def gmm_1d(
    samples,
    weights,
    means,
    stdvs,
    axis=None,
    hist_color=None,
    line_color="black",
    bins=40,
):
    """Histogram with 1D GMM density overlay."""
    if axis is None:
        _, axis = plt.subplots()
    if hist_color is None:
        hist_color = DEFAULT_PALETTE.neutral
    samples = np.asarray(samples).flatten()
    histogram(samples, axis, bins=bins, color=hist_color, alpha=0.6)
    x_min, x_max = samples.min() - 1.0, samples.max() + 1.0
    x_grid = np.linspace(x_min, x_max, 400)
    weights = np.asarray(weights)
    means = np.asarray(means).flatten()
    stdvs = np.asarray(stdvs).flatten()
    density_vals = np.zeros_like(x_grid)
    for w, m, s in zip(weights, means, stdvs):
        density_vals += w * stats.norm(m, s).pdf(x_grid)
    axis.plot(x_grid, density_vals, color=line_color, linewidth=2)
    return axis


def gmm_2d(
    samples,
    weights,
    means,
    covariances,
    axis=None,
    scatter_color=None,
    contour_color="black",
    levels=10,
):
    """Scatter plot with 2D GMM contours."""
    if axis is None:
        _, axis = plt.subplots()
    if scatter_color is None:
        scatter_color = DEFAULT_PALETTE.secondary
    samples = np.asarray(samples)
    scatter(
        samples[:, 0], samples[:, 1], axis, s=8, alpha=0.35, color=scatter_color
    )
    x_min, x_max = samples[:, 0].min() - 1.0, samples[:, 0].max() + 1.0
    y_min, y_max = samples[:, 1].min() - 1.0, samples[:, 1].max() + 1.0
    x_grid = np.linspace(x_min, x_max, 120)
    y_grid = np.linspace(y_min, y_max, 120)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    weights = np.asarray(weights)
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    density_vals = np.zeros(grid.shape[0])
    for w, m, cov in zip(weights, means, covariances):
        density_vals += w * stats.multivariate_normal(m, cov).pdf(grid)
    Z = density_vals.reshape(X.shape)
    axis.contour(X, Y, Z, levels=levels, colors=contour_color)
    axis.set_aspect("equal", adjustable="box")
    return axis


# ---------------------------------------------------------------------------
# Group plotting functions
# ---------------------------------------------------------------------------
def group_scatter(
    x, y, groups, axis=None, colors=None, alpha=0.6, s=20, labels=None
):
    """Scatter plot with different colors per group."""
    if axis is None:
        _, axis = plt.subplots()
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    unique_groups = np.unique(groups)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(unique_groups))]
    for i, g in enumerate(unique_groups):
        mask = groups == g
        label = labels[i] if labels is not None else f"group {g}"
        axis.scatter(
            x[mask], y[mask], color=colors[i], alpha=alpha, s=s, label=label
        )
    return axis


def group_lines(
    x, y_samples, groups, axis=None, colors=None, alpha=0.1, num_lines=50
):
    """Plot lines colored by group from posterior samples."""
    if axis is None:
        _, axis = plt.subplots()
    x = np.asarray(x)
    y_samples = np.asarray(y_samples)
    unique_groups = np.unique(groups)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(unique_groups))]
    num_available = y_samples.shape[0]
    num_to_plot = min(num_lines, num_available)
    indices = np.random.choice(num_available, num_to_plot, replace=False)
    for idx in indices:
        for i, g in enumerate(unique_groups):
            axis.plot(x, y_samples[idx, g], alpha=alpha, color=colors[i])
    return axis


# ---------------------------------------------------------------------------
# Predictive plots
# ---------------------------------------------------------------------------
def prior_predictive(
    x, y_samples, axis=None, color=None, alpha=0.1, num_lines=20
):
    """Plot samples from prior predictive distribution."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.secondary
    y_samples = np.asarray(y_samples)
    if y_samples.ndim == 1:
        y_samples = y_samples[np.newaxis, :]
    num_available = y_samples.shape[0]
    num_to_plot = min(num_lines, num_available)
    indices = np.random.choice(num_available, num_to_plot, replace=False)
    for i in indices:
        axis.plot(x, y_samples[i], alpha=alpha, color=color)
    return axis


def posterior_predictive(
    x,
    y_samples,
    data_x=None,
    data_y=None,
    axis=None,
    color=None,
    alpha=0.1,
    num_lines=50,
    data_color="red",
    show_mean=True,
    show_interval=True,
    interval=0.95,
):
    """Plot posterior predictive with optional mean and credible interval."""
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    if color is None:
        color = DEFAULT_PALETTE.secondary
    y_samples = np.asarray(y_samples)
    if y_samples.ndim == 1:
        y_samples = y_samples[np.newaxis, :]
    num_available = y_samples.shape[0]
    num_to_plot = min(num_lines, num_available)
    indices = np.random.choice(num_available, num_to_plot, replace=False)
    for i in indices:
        axis.plot(x, y_samples[i], alpha=alpha, color=color)
    if show_mean:
        y_mean = y_samples.mean(axis=0)
        axis.plot(x, y_mean, color="black", linewidth=2, label="mean")
    if show_interval:
        lower_p = (1 - interval) / 2
        upper_p = 1 - lower_p
        y_lower = np.percentile(y_samples, lower_p * 100, axis=0)
        y_upper = np.percentile(y_samples, upper_p * 100, axis=0)
        axis.fill_between(
            x,
            y_lower,
            y_upper,
            alpha=0.2,
            color=color,
            label=f"{int(interval*100)}% CI",
        )
    if data_x is not None and data_y is not None:
        axis.scatter(
            data_x,
            data_y,
            color=data_color,
            alpha=0.8,
            s=20,
            label="data",
            zorder=5,
        )
    return fig, axis


# ---------------------------------------------------------------------------
# MCMC diagnostics and panels
# ---------------------------------------------------------------------------
def posterior_panel(samples_dict, true_values=None, bins=50, figsize=None):
    """Grid of posterior histograms for each parameter."""
    names = list(samples_dict.keys())
    num_params = len(names)
    ncols = min(3, num_params)
    nrows = math.ceil(num_params / ncols)
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_params == 1:
        axes = np.array([[axes]])
    else:
        axes = np.atleast_2d(axes)
    for idx, name in enumerate(names):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        samples = np.asarray(samples_dict[name]).flatten()
        histogram(samples, axis=ax, bins=bins)
        if true_values is not None and name in true_values:
            vline(true_values[name], axis=ax, color="red", label="true")
        ax.set_xlabel(name)
        clean(ax)
    for idx in range(num_params, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis("off")
    plt.tight_layout()
    return fig, axes


def trace_panel(samples_dict, figsize=None, title=None):
    """Grid of trace + density for each parameter."""
    names = list(samples_dict.keys())
    num_params = len(names)
    if figsize is None:
        figsize = (10, 2.5 * num_params)
    fig, axes = plt.subplots(num_params, 2, figsize=figsize)
    if num_params == 1:
        axes = axes[np.newaxis, :]
    for idx, name in enumerate(names):
        samples = np.asarray(samples_dict[name])
        trace_lines(samples, axis=axes[idx, 0])
        axes[idx, 0].set_ylabel(name)
        axes[idx, 0].set_xlabel("iteration")
        clean(axes[idx, 0])
        density(samples.flatten(), axis=axes[idx, 1])
        axes[idx, 1].set_xlabel(name)
        axes[idx, 1].set_ylabel("density")
        clean(axes[idx, 1])
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, axes


def diagnostics(acceptance_rates, axis=None, color=None):
    """Bar chart of acceptance rates per chain."""
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 4))
    else:
        fig = axis.get_figure()
    if color is None:
        color = DEFAULT_PALETTE.primary
    acceptance_rates = np.asarray(acceptance_rates)
    if acceptance_rates.ndim == 2:
        acceptance_rates = acceptance_rates.mean(axis=0)
    acceptance_rates = acceptance_rates.flatten()
    num_chains = len(acceptance_rates)
    x = np.arange(num_chains)
    axis.bar(x, acceptance_rates, color=color, alpha=0.8)
    axis.set_xlabel("chain")
    axis.set_ylabel("acceptance rate")
    axis.set_xticks(x)
    mean_rate = acceptance_rates.mean()
    hline(mean_rate, axis=axis, color="red", label=f"mean: {mean_rate:.3f}")
    axis.legend(frameon=False)
    clean(axis)
    return fig, axis


# ---------------------------------------------------------------------------
# Enhanced visualizations
# ---------------------------------------------------------------------------
def corner(samples_dict, true_values=None, bins=30, figsize=None):
    """Corner plot showing pairwise relationships and marginals."""
    names = list(samples_dict.keys())
    num_params = len(names)
    if figsize is None:
        figsize = (2.5 * num_params, 2.5 * num_params)
    fig, axes = plt.subplots(num_params, num_params, figsize=figsize)
    samples_arrays = {
        k: np.asarray(v).flatten() for k, v in samples_dict.items()
    }
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                histogram(
                    samples_arrays[name_i],
                    axis=ax,
                    bins=bins,
                    color=DEFAULT_PALETTE.primary,
                )
                if true_values and name_i in true_values:
                    vline(true_values[name_i], ax, color="red")
            else:
                scatter(
                    samples_arrays[name_j],
                    samples_arrays[name_i],
                    axis=ax,
                    s=1,
                    alpha=0.1,
                )
                if true_values:
                    if name_j in true_values:
                        vline(true_values[name_j], ax, color="red")
                    if name_i in true_values:
                        hline(true_values[name_i], ax, color="red")
            if i == num_params - 1:
                ax.set_xlabel(name_j)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(name_i)
            else:
                ax.set_yticklabels([])
            hide_spines(ax, "box")
    plt.tight_layout()
    return fig, axes


def forest_plot(
    names,
    means,
    lower,
    upper,
    true_values=None,
    axis=None,
    color=None,
    true_color="red",
):
    """Forest plot with credible intervals."""
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 0.5 * len(names) + 1))
    else:
        fig = axis.get_figure()
    if color is None:
        color = DEFAULT_PALETTE.primary
    names = list(names)
    means = np.asarray(means)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y_pos = np.arange(len(names))
    axis.errorbar(
        means,
        y_pos,
        xerr=[means - lower, upper - means],
        fmt="o",
        color=color,
        capsize=4,
        capthick=2,
        markersize=8,
    )
    if true_values is not None:
        true_values = np.asarray(true_values)
        axis.scatter(
            true_values,
            y_pos,
            marker="x",
            color=true_color,
            s=100,
            zorder=5,
            label="true",
        )
    axis.set_yticks(y_pos)
    axis.set_yticklabels(names)
    axis.set_xlabel("value")
    vline(0, axis, color="gray", linestyle=":", linewidth=1)
    hide_spines(axis, "box")
    axis.invert_yaxis()
    return fig, axis


def prior_posterior_comparison(
    prior_samples,
    posterior_samples,
    name,
    axis=None,
    prior_color=None,
    posterior_color=None,
    true_value=None,
):
    """Compare prior and posterior distributions."""
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    if prior_color is None:
        prior_color = DEFAULT_PALETTE.neutral
    if posterior_color is None:
        posterior_color = DEFAULT_PALETTE.primary
    density(
        prior_samples,
        axis,
        color=prior_color,
        fill=True,
        alpha=0.3,
        label="prior",
    )
    density(
        posterior_samples,
        axis,
        color=posterior_color,
        fill=True,
        alpha=0.5,
        label="posterior",
    )
    if true_value is not None:
        vline(true_value, axis, color="red", label="true")
    axis.set_xlabel(name)
    axis.set_ylabel("density")
    axis.legend(frameon=False)
    clean(axis)
    return fig, axis


def shrinkage_plot(
    group_estimates,
    pooled_estimate,
    true_values=None,
    axis=None,
    group_color=None,
    pooled_color="red",
    true_color="black",
):
    """Visualize hierarchical shrinkage effect."""
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    if group_color is None:
        group_color = DEFAULT_PALETTE.primary
    group_estimates = np.asarray(group_estimates)
    num_groups = len(group_estimates)
    y_pos = np.arange(num_groups)
    axis.scatter(
        group_estimates,
        y_pos,
        color=group_color,
        s=100,
        label="group estimate",
        zorder=3,
    )
    vline(
        pooled_estimate,
        axis,
        color=pooled_color,
        linestyle="-",
        linewidth=2,
        label="pooled",
    )
    for i, est in enumerate(group_estimates):
        axis.plot(
            [est, pooled_estimate],
            [i, i],
            color="gray",
            alpha=0.5,
            linestyle=":",
        )
    if true_values is not None:
        true_values = np.asarray(true_values)
        axis.scatter(
            true_values,
            y_pos,
            marker="x",
            color=true_color,
            s=80,
            label="true",
            zorder=4,
        )
    axis.set_yticks(y_pos)
    axis.set_yticklabels([f"group {i}" for i in range(num_groups)])
    axis.set_xlabel("estimate")
    axis.legend(frameon=False, loc="best")
    hide_spines(axis, "box")
    return fig, axis


# ---------------------------------------------------------------------------
# Bayesian inverse graphics helpers
# ---------------------------------------------------------------------------
def _require_arviz():
    if az is None:
        raise ImportError("arviz is required for this plotting function.")


def _require_ternary():
    if ternary is None or simplex_iterator is None:
        raise ImportError(
            "python-ternary is required for this plotting function."
        )


def _get_figure(axes):
    if isinstance(axes, np.ndarray):
        axis = axes.ravel()[0]
    else:
        axis = axes
    return axis.get_figure()


def _compute_limits(samples, true_mean, half_size):
    data_max = samples.max()
    data_min = samples.min()
    true_min = true_mean - half_size
    true_max = true_mean + half_size
    limit_min = np.max([true_min, data_min])
    limit_max = np.min([true_max, data_max])
    return [limit_min, limit_max]


def _pair_kde(x, y, axis, cmap="viridis", levels=8):
    values = np.vstack([x, y])
    kde = stats.gaussian_kde(values)
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_grid = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(grid).reshape(X.shape)
    axis.contourf(X, Y, Z, levels=levels, cmap=cmap)
    axis.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.5, alpha=0.5)


def plot_trace(data, var_names=None):
    # TODO let's not use arviz. Maybe we can re-use the previous trace plots without arviz.
    _require_arviz()
    axes = az.plot_trace(data, var_names=var_names)
    figure = _get_figure(axes)
    return figure, axes


def plot_trace_variable(data, variable_name):
    return plot_trace(data, var_names=[variable_name])


def plot_shift_posterior(
    trace, true_point=None, name="shift", cmap="viridis", marker_color="red"
):
    x_samples = np.asarray(trace[name])[:, :, 0].flatten()
    y_samples = np.asarray(trace[name])[:, :, 1].flatten()
    figure, axes = plt.subplots(2, 2, figsize=(6, 6))
    _pair_kde(x_samples, y_samples, axes[1, 0], cmap=cmap)
    if true_point is not None:
        x_true, y_true = true_point
        axes[1, 0].scatter(x_true, y_true, s=30, marker="*", c=marker_color)
        half_box = 0.05
        x_limits = _compute_limits(x_samples, x_true, half_box)
        y_limits = _compute_limits(y_samples, y_true, half_box)
        axes[1, 0].set_xlim(x_limits)
        axes[1, 0].set_ylim(y_limits)
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[1, 1].axis("off")
    hide_spines(axes[1, 0], "box")
    return figure, axes


def plot_theta_posterior(samples, bins=100, color=None):
    if color is None:
        color = DEFAULT_PALETTE.primary
    values = np.asarray(samples).flatten()
    values = (values + np.pi) % (2 * np.pi) - np.pi
    figure, axis = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
    counts, bin_edges = np.histogram(values, bins=bins)
    area = counts / values.size
    radius = (area / np.pi) ** 0.5
    widths = np.diff(bin_edges)
    axis.bar(
        bin_edges[:-1],
        radius,
        align="edge",
        width=widths,
        color=color,
        edgecolor=color,
        linewidth=1,
    )
    axis.set_yticklabels([])
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    return figure, axis


def plot_shift_posteriors(data):
    _require_arviz()
    axes = az.plot_posterior(data, var_names=["shift"])
    return _get_figure(axes), axes


def plot_shift_x_posterior(trace):
    _require_arviz()
    data = {"shift_x": np.asarray(trace["shift"])[:, :, 0]}
    axes = az.plot_posterior(data, var_names=["shift_x"])
    return _get_figure(axes), axes


def plot_shift_y_posterior(trace):
    _require_arviz()
    data = {"shift_y": np.asarray(trace["shift"])[:, :, 1]}
    axes = az.plot_posterior(data, var_names=["shift_y"])
    return _get_figure(axes), axes


def plot_scale_posterior(data):
    _require_arviz()
    axes = az.plot_posterior(data, var_names=["scale"])
    return _get_figure(axes), axes


def plot_scale_x_posterior(trace):
    _require_arviz()
    data = {"scale_x": np.asarray(trace["scale"])[:, :, 0]}
    axes = az.plot_posterior(data, var_names=["scale_x"])
    return _get_figure(axes), axes


def plot_scale_y_posterior(trace):
    _require_arviz()
    data = {"scale_y": np.asarray(trace["scale"])[:, :, 1]}
    axes = az.plot_posterior(data, var_names=["scale_y"])
    return _get_figure(axes), axes


def plot_scale_z_posterior(trace):
    _require_arviz()
    data = {"scale_z": np.asarray(trace["scale"])[:, :, 2]}
    axes = az.plot_posterior(data, var_names=["scale_z"])
    return _get_figure(axes), axes


def plot_color_posterior(data):
    _require_arviz()
    axes = az.plot_posterior(data, var_names=["color"])
    return _get_figure(axes), axes


def plot_ambient_posterior(data):
    _require_arviz()
    axes = az.plot_posterior(data, var_names=["ambient"])
    return _get_figure(axes), axes


def plot_diffuse_posterior(data):
    _require_arviz()
    axes = az.plot_posterior(data, var_names=["diffuse"])
    return _get_figure(axes), axes


def plot_specular_posterior(data):
    _require_arviz()
    axes = az.plot_posterior(data, var_names=["specular"])
    return _get_figure(axes), axes


def plot_shininess_posterior(data):
    _require_arviz()
    axes = az.plot_posterior(data, var_names=["shininess"])
    return _get_figure(axes), axes


def plot_classes_posterior(trace):
    _require_arviz()
    classes = np.asarray(trace["classes"])
    data = {
        "class_0": classes[:, :, 0],
        "class_1": classes[:, :, 1],
        "class_2": classes[:, :, 2],
    }
    axes = az.plot_posterior(data)
    return _get_figure(axes), axes


def _compute_pairwise_distances(x, y):
    n = x.shape[0]
    m = y.shape[0]
    x = np.tile(np.expand_dims(x, 1), [1, m, 1])
    y = np.tile(np.expand_dims(y, 0), [n, 1, 1])
    return np.mean(np.power(x - y, 2), 2)


def _wrap_density(scale, corner_density):
    density = {}
    for (i, j, k), value in zip(simplex_iterator(scale), corner_density):
        density[(i, j)] = value
    return density


def _compute_corner_density(corners, probabilities):
    distances = _compute_pairwise_distances(corners, probabilities)
    closest_corners = np.argmin(distances, axis=0)
    corner_density = []
    for corner_arg in range(len(corners)):
        density = np.sum(corner_arg == closest_corners)
        corner_density.append(density)
    return corner_density


def _plot_dirichlet_colorbar(axis, density, cmap, shrink=0.80):
    colorbar_kwargs = {"shrink": shrink, "format": "%.0e"}
    vmin = min(density.values())
    vmax = max(density.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar = plt.colorbar(sm, ax=axis, **colorbar_kwargs)
    colorbar.ax.get_yaxis().labelpad = 15
    colorbar.set_label("frequency", rotation=90, fontsize=15)
    colorbar.ax.tick_params(labelsize=10)
    return colorbar


def _plot_dirichlet_ticks(axis, scale, fontsize=10):
    ticks = np.linspace(0, 1, 11).tolist()
    locations = np.linspace(0, scale, 11).tolist()
    ternary.lines.ticks(
        axis,
        scale,
        ticks=ticks,
        locations=locations,
        clockwise=False,
        axis="blr",
        tick_formats=" %.1f",
        offset=0.0180,
        fontsize=fontsize,
    )
    ternary.plotting.clear_matplotlib_ticks(ax=axis, axis="both")


def _plot_dirichlet_point_estimate(axis, scale, point_estimate, color):
    point_a = np.array([0, scale - point_estimate[0], point_estimate[0]])
    point_b = np.array([scale - point_estimate[1], point_estimate[1], 0])
    point_c = np.array([point_estimate[2], 0, scale - point_estimate[2]])

    ternary.line(axis, point_a, point_estimate, "012", color=color)
    ternary.line(axis, point_b, point_estimate, "012", color=color)
    ternary.line(axis, point_c, point_estimate, "012", color=color)

    x, y, z = point_estimate / scale
    point_estimate = np.reshape(point_estimate, (1, 3))
    ternary.scatter(
        points=point_estimate,
        ax=axis,
        marker="o",
        color=color,
        label=f"mean point: [{x:.2f}, {y:.2f}, {z:.2f}]",
    )
    axis.legend(prop={"size": 10}, loc=(0.6, 0.95), frameon=False)
    axis.axis("off")


def _plot_dirichlet_labels(axis, scale):
    _, tax = ternary.figure(axis, scale=scale)
    tax.left_axis_label(r"$\kappa_{\mathrm{sphere}}$", offset=0.18)
    tax.right_axis_label(r"$\kappa_{\mathrm{cube}}$", offset=0.18)
    tax.bottom_axis_label(r"$\kappa_{\mathrm{cylinder}}$", offset=0.1)


def plot_dirichlet_posterior(trace, scale=20, cmap="viridis"):
    _require_ternary()
    figure, axis = plt.subplots()
    corners = list(simplex_iterator(scale=scale))
    corners = np.array(corners) / scale
    probabilities = np.asarray(trace["classes"]).reshape(-1, 3)
    corner_density = _compute_corner_density(corners, probabilities)
    density = _wrap_density(scale, corner_density)
    ternary.heatmap(
        density, scale, ax=axis, cmap=cmap, colorbar=False, style="h"
    )
    _plot_dirichlet_colorbar(axis, density, cmap)
    _plot_dirichlet_ticks(axis, scale, fontsize=10)
    point_estimate = np.mean(probabilities, axis=0) * scale
    _plot_dirichlet_point_estimate(
        axis, scale, point_estimate, DANDELION.primary
    )
    _plot_dirichlet_labels(axis, scale)
    return figure, axis


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------
def save(figure, filepath):
    """Save figure with tight bounding box."""
    figure.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(figure)


def show():
    """Display current figure."""
    plt.show()


def subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    """Create subplots with sensible defaults."""
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
