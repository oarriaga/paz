import math
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

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
def density(samples, axis=None, color=None, fill=True, alpha=0.3, label=None,
            linewidth=2.0):
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


def histogram(samples, axis=None, bins=50, density_norm=True, color=None,
              alpha=0.7, label=None, edgecolor="white"):
    """Histogram with optional density normalization."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    samples = np.asarray(samples).flatten()
    axis.hist(samples, bins=bins, density=density_norm, alpha=alpha,
              color=color, label=label, edgecolor=edgecolor)
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
        axis.plot(samples[:, chain], alpha=alpha, linewidth=linewidth,
                  color=colors[chain])
    return axis


def scatter(x, y, axis=None, alpha=0.5, s=10, color=None, label=None):
    """Scatter plot."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.scatter(x, y, alpha=alpha, s=s, color=color, label=label)
    return axis


def line(x, y, axis=None, color=None, linewidth=2.0, linestyle="-", label=None,
         alpha=1.0):
    """Simple line plot."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle,
              label=label, alpha=alpha)
    return axis


def fill_between(x, y_lower, y_upper, axis=None, color=None, alpha=0.2,
                 label=None):
    """Shaded region between two curves."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.fill_between(x, y_lower, y_upper, alpha=alpha, color=color,
                      label=label)
    return axis


def line_with_band(x, y_mean, y_std=None, axis=None, color=None, alpha=0.2,
                   label=None, linewidth=2.0):
    """Line plot with optional shaded uncertainty band."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.plot(x, y_mean, color=color, linewidth=linewidth, label=label)
    if y_std is not None:
        axis.fill_between(x, y_mean - y_std, y_mean + y_std,
                          alpha=alpha, color=color)
    return axis


def regression_lines(x, slopes, intercepts, axis=None, alpha=0.1, color=None,
                     num_lines=50):
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


def contour(samples_x, samples_y, axis=None, levels=10, cmap="viridis",
            fill=True):
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
    axis.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth,
                 label=label)
    return axis


def hline(y, axis=None, color=None, linestyle="--", linewidth=1.5, label=None):
    """Add horizontal line."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.secondary
    axis.axhline(y, color=color, linestyle=linestyle, linewidth=linewidth,
                 label=label)
    return axis


def bar(x, heights, axis=None, color=None, alpha=0.8, width=0.8, label=None):
    """Bar plot."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.primary
    axis.bar(x, heights, color=color, alpha=alpha, width=width, label=label)
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
    axis.scatter(theoretical_quantiles, sorted_samples, alpha=0.3, s=5,
                 color=color)
    lims = [
        min(theoretical_quantiles.min(), sorted_samples.min()),
        max(theoretical_quantiles.max(), sorted_samples.max()),
    ]
    axis.plot(lims, lims, color=line_color, linestyle="--", linewidth=2)
    set_labels(axis, x="theoretical quantiles", y="sample quantiles")
    return axis


def compare_densities(x_range, densities, labels, axis=None, colors=None,
                      linestyles=None, linewidth=2.0):
    """Plot multiple density curves on the same axis."""
    if axis is None:
        _, axis = plt.subplots()
    if colors is None:
        colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]
    if linestyles is None:
        linestyles = ["-", "--", ":", "-.", "-"]
    for i, (density_vals, label) in enumerate(zip(densities, labels)):
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        axis.plot(x_range, density_vals, color=color, linestyle=ls,
                  linewidth=linewidth, label=label)
    return axis


def discrete_posterior(support, probabilities, axis=None, true_value=None,
                       color=None, true_color="tab:green"):
    """Bar plot for discrete posterior probabilities."""
    if axis is None:
        _, axis = plt.subplots()
    if color is None:
        color = DEFAULT_PALETTE.accent
    support = np.asarray(support)
    probabilities = np.asarray(probabilities)
    axis.bar(support, probabilities, color=color, alpha=0.7, width=0.8)
    if true_value is not None:
        vline(true_value, axis, color=true_color, linestyle="--",
              label="true value")
    set_labels(axis, y="probability")
    set_limits(axis, y=(0, 1.05))
    return axis


# ---------------------------------------------------------------------------
# GMM plotting functions
# ---------------------------------------------------------------------------
def gmm_1d(samples, weights, means, stdvs, axis=None, hist_color=None,
           line_color="black", bins=40):
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


def gmm_2d(samples, weights, means, covariances, axis=None, scatter_color=None,
           contour_color="black", levels=10):
    """Scatter plot with 2D GMM contours."""
    if axis is None:
        _, axis = plt.subplots()
    if scatter_color is None:
        scatter_color = DEFAULT_PALETTE.secondary
    samples = np.asarray(samples)
    scatter(samples[:, 0], samples[:, 1], axis, s=8, alpha=0.35,
            color=scatter_color)
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
def group_scatter(x, y, groups, axis=None, colors=None, alpha=0.6, s=20,
                  labels=None):
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
        axis.scatter(x[mask], y[mask], color=colors[i], alpha=alpha, s=s,
                     label=label)
    return axis


def group_lines(x, y_samples, groups, axis=None, colors=None, alpha=0.1,
                num_lines=50):
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
def prior_predictive(x, y_samples, axis=None, color=None, alpha=0.1,
                     num_lines=20):
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


def posterior_predictive(x, y_samples, data_x=None, data_y=None, axis=None,
                         color=None, alpha=0.1, num_lines=50,
                         data_color="red", show_mean=True, show_interval=True,
                         interval=0.95):
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
        axis.fill_between(x, y_lower, y_upper, alpha=0.2, color=color,
                          label=f"{int(interval*100)}% CI")
    if data_x is not None and data_y is not None:
        axis.scatter(data_x, data_y, color=data_color, alpha=0.8, s=20,
                     label="data", zorder=5)
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
    samples_arrays = {k: np.asarray(v).flatten() for k, v in samples_dict.items()}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                histogram(samples_arrays[name_i], axis=ax, bins=bins,
                          color=DEFAULT_PALETTE.primary)
                if true_values and name_i in true_values:
                    vline(true_values[name_i], ax, color="red")
            else:
                scatter(samples_arrays[name_j], samples_arrays[name_i],
                        axis=ax, s=1, alpha=0.1)
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


def forest_plot(names, means, lower, upper, true_values=None, axis=None,
                color=None, true_color="red"):
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
    axis.errorbar(means, y_pos, xerr=[means - lower, upper - means],
                  fmt="o", color=color, capsize=4, capthick=2, markersize=8)
    if true_values is not None:
        true_values = np.asarray(true_values)
        axis.scatter(true_values, y_pos, marker="x", color=true_color, s=100,
                     zorder=5, label="true")
    axis.set_yticks(y_pos)
    axis.set_yticklabels(names)
    axis.set_xlabel("value")
    vline(0, axis, color="gray", linestyle=":", linewidth=1)
    hide_spines(axis, "box")
    axis.invert_yaxis()
    return fig, axis


def prior_posterior_comparison(prior_samples, posterior_samples, name,
                               axis=None, prior_color=None,
                               posterior_color=None, true_value=None):
    """Compare prior and posterior distributions."""
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    if prior_color is None:
        prior_color = DEFAULT_PALETTE.neutral
    if posterior_color is None:
        posterior_color = DEFAULT_PALETTE.primary
    density(prior_samples, axis, color=prior_color, fill=True, alpha=0.3,
            label="prior")
    density(posterior_samples, axis, color=posterior_color, fill=True,
            alpha=0.5, label="posterior")
    if true_value is not None:
        vline(true_value, axis, color="red", label="true")
    axis.set_xlabel(name)
    axis.set_ylabel("density")
    axis.legend(frameon=False)
    clean(axis)
    return fig, axis


def shrinkage_plot(group_estimates, pooled_estimate, true_values=None,
                   axis=None, group_color=None, pooled_color="red",
                   true_color="black"):
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
    axis.scatter(group_estimates, y_pos, color=group_color, s=100,
                 label="group estimate", zorder=3)
    vline(pooled_estimate, axis, color=pooled_color, linestyle="-",
          linewidth=2, label="pooled")
    for i, est in enumerate(group_estimates):
        axis.plot([est, pooled_estimate], [i, i], color="gray", alpha=0.5,
                  linestyle=":")
    if true_values is not None:
        true_values = np.asarray(true_values)
        axis.scatter(true_values, y_pos, marker="x", color=true_color, s=80,
                     label="true", zorder=4)
    axis.set_yticks(y_pos)
    axis.set_yticklabels([f"group {i}" for i in range(num_groups)])
    axis.set_xlabel("estimate")
    axis.legend(frameon=False, loc="best")
    hide_spines(axis, "box")
    return fig, axis


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
