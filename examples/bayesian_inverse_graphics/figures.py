import math
from pathlib import Path

import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats

import paz
from paz.utils import plot

DANDELION = plot.DANDELION.primary
GRAY = "0.55"
COMPONENT_LABELS = {
    "shift": ["$x$ [m]", "$y$ [m]"],
    "theta": [r"$\theta$ [rad]"],
    "scale": ["$s_x$ [m]", "$s_y$ [m]", "$s_z$ [m]"],
    "color": ["$c_r$", "$c_g$", "$c_b$"],
    "ambient": ["$k^a$"],
    "diffuse": ["$k^d$"],
    "specular": ["$k^s$"],
    "shininess": [r"$\alpha$"],
}


def configure():
    plot.configure(fontsize=9, latex=True, linewidth=1.2)
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7


def save(figure, directory, name):
    for extension in ["pdf", "png"]:
        filepath = Path(directory) / f"{name}.{extension}"
        figure.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(figure)


def write_posterior_figures(directory, trace, priors, true_values, images):
    panels = build_posterior_panels(trace, priors, true_values)
    write_panels(panels, directory)
    sheet = plot_sheet({**build_image_panels(*images), **panels}, 3, 2.1)
    save(sheet, directory, "posterior_sheet")
    save(plot_trace(trace), directory, "trace")


def write_prior_figures(directory, priors, samples, inverses, values, losses):
    """Writes prior and bijector figures; `inverses` holds before and after."""
    prior_panels = build_prior_panels(priors, samples, values)
    write_panels(prior_panels, directory)
    save(plot_sheet(prior_panels, 3, 1.7), directory, "priors")
    fit_panels = build_bijector_panels(*inverses)
    write_panels(fit_panels, directory)
    save(plot_sheet(fit_panels, 4, 1.5), directory, "bijector_fits")
    save(plot_losses(losses), directory, "bijector_losses")


def write_feature_figures(directory, images, maps, selected, errors):
    """Writes the feature maps of each example and the feature errors.

    `images` is `(2, N, H, W, 3)` and each of `maps` is `(2, N, h, w)`, with
    the observations first and the renders second.
    """
    for example_arg in range(images.shape[1]):
        example_maps = [feature_maps[:, example_arg] for feature_maps in maps]
        figure_args = (images[:, example_arg], example_maps, selected)
        name = f"feature_maps_{example_arg:02d}"
        save(plot_feature_maps(*figure_args), directory, name)
    save(plot_feature_errors(errors, selected), directory, "feature_errors")


def plot_feature_maps(images, maps, selected):
    num_columns = len(maps) + 1
    H, W = images.shape[1:3]
    size = (6.5, 3 * (6.5 / num_columns) * (H / W) + 0.3)
    grid = dict(figsize=size, layout="constrained")
    figure, axes = plt.subplots(3, num_columns, **grid)
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0)
    draw_image_column(axes[:, 0], images)
    for column, feature_maps, feature in zip(axes.T[1:], maps, selected):
        draw_feature_column(column, feature_maps)
        column[0].set_title(build_feature_label(*feature), fontsize=8)
    for axis, label in zip(axes[:, 0], ["observation", "render", "difference"]):
        axis.set_ylabel(label, fontsize=7)
    return figure


def draw_image_column(axes, images):
    difference = np.abs(images[0] - images[1]).mean(axis=-1)
    axes[0].imshow(images[0])
    axes[1].imshow(images[1])
    axes[2].imshow(difference, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("image", fontsize=8)
    hide_frames(axes)


def draw_feature_column(axes, feature_maps):
    # One colour scale per feature keeps the three rows comparable.
    scale = dict(cmap="viridis", vmin=0.0, vmax=feature_maps.max())
    difference = np.abs(feature_maps[0] - feature_maps[1])
    for axis, feature_map in zip(axes, [*feature_maps, difference]):
        axis.imshow(feature_map, **scale)
    hide_frames(axes)


def hide_frames(axes):
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)


def build_feature_label(layer, channel):
    block = ",".join(filter(str.isdigit, layer))
    return rf"$\phi^{{{block}}}_{{{channel}}}$"


def build_layer_label(layer):
    return layer.replace("block", "").replace("_conv", ".")


def plot_feature_errors(errors, selected):
    figure, axis = plt.subplots(figsize=(4.2, 2.6), layout="constrained")
    errors = raise_zero_errors(errors)
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(errors)))
    for (layer, values), color in zip(errors.items(), colors):
        ranks = np.arange(1, len(values) + 1)
        axis.plot(ranks, np.sort(values), color=color, linewidth=1.0)
        axis.plot([], [], color=color, label=build_layer_label(layer))
    points = [compute_rank_point(errors[layer], arg) for layer, arg in selected]
    dots = dict(color=DANDELION, markersize=3, linestyle="none", zorder=3)
    axis.plot(*np.array(points).T, "o", label="selected", **dots)
    axis.set_xscale("log")
    axis.set_yscale("log")
    plot.clean(axis, "box")
    labels = dict(x="feature rank within layer", y="mean squared error")
    plot.set_labels(axis, labelpad=4, **labels)
    axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return figure


def raise_zero_errors(errors, floor=1e-3):
    # A log axis would drop the features that match with exactly zero error.
    raised = {}
    for layer, values in errors.items():
        raised[layer] = np.maximum(values, floor)
    return raised


def compute_rank_point(values, channel):
    rank = np.sum(values < values[channel]) + 1
    return rank, values[channel]


def plot_ground_shift(samples, image, floor_image, project):
    """Draws the location posterior on the ground under a see-through object.

    `project` maps ground locations `(N, 2)` to pixels `(N, 2)` of `image`.
    """
    x, y, mass = compute_joint_mass(*samples.T)
    grid = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    pixels = np.asarray(project(jp.array(grid.reshape(-1, 2))))
    opacity = compute_object_opacity(image, floor_image)
    layers = (floor_image, image, opacity, pixels.reshape(grid.shape), mass)
    window = compute_zoom_window(opacity)
    figure = plt.figure(figsize=(6.5, 2.5), layout="constrained")
    cells = figure.add_gridspec(1, 2)
    whole_axis = figure.add_subplot(cells[0])
    draw_ground_shift(whole_axis, *layers)
    draw_zoom_frame(whole_axis, window)
    zoom_axis = figure.add_subplot(cells[1])
    draw_ground_shift(zoom_axis, *layers)
    zoom_axis.axis(window)
    zoom_axis.axis("off")
    return figure


def compute_object_opacity(image, floor_image, threshold=0.08):
    difference = np.abs(np.asarray(image) - np.asarray(floor_image))
    return np.clip(difference.max(axis=-1) / threshold, 0.0, 1.0)


def compute_zoom_window(opacity, padding=0.25, aspect=4 / 3):
    rows, columns = np.nonzero(opacity > 0.5)
    center_x = (columns.min() + columns.max()) / 2
    center_y = (rows.min() + rows.max()) / 2
    height = max(np.ptp(rows), np.ptp(columns) / aspect) * (1 + padding)
    half_x, half_y = aspect * height / 2, height / 2
    horizontal = (center_x - half_x, center_x + half_x)
    return (*horizontal, center_y + half_y, center_y - half_y)


def draw_ground_shift(axis, floor_image, image, opacity, pixels, mass):
    levels = compute_highest_density_levels(mass)
    u, v = pixels[..., 0], pixels[..., 1]
    axis.imshow(floor_image)
    axis.contourf(u, v, mass, levels=[*levels, mass.max()], cmap="viridis")
    axis.contour(u, v, mass, levels=levels, colors="w", linewidths=0.3)
    axis.imshow(np.dstack([image, 0.6 * opacity]))
    axis.axis("off")


def draw_zoom_frame(axis, window):
    left, right, bottom, top = window
    size = (right - left, bottom - top)
    style = dict(fill=False, edgecolor=DANDELION, linewidth=0.8)
    axis.add_patch(plt.Rectangle((left, top), *size, **style))


def write_panels(panels, directory):
    for name, (draw, *args) in panels.items():
        save(plot_panel(draw, *args), directory, name)


def plot_panel(draw, *args):
    figure = plt.figure(figsize=(3.0, 2.4), layout="constrained")
    draw(figure, figure.add_gridspec(1, 1)[0], *args)
    return figure


def plot_sheet(panels, num_columns, row_height):
    num_rows = math.ceil(len(panels) / num_columns)
    size = (6.5, row_height * num_rows)
    figure = plt.figure(figsize=size, layout="constrained")
    grid = figure.add_gridspec(num_rows, num_columns)
    for arg, (draw, *args) in enumerate(panels.values()):
        draw(figure, grid[arg], *args)
    return figure


def plot_losses(losses):
    figure, axis = plt.subplots(figsize=(3.0, 2.4), layout="constrained")
    for name, values in losses.items():
        axis.plot(values, label=name, linewidth=0.8)
    finish_axis(axis, "step", "negative log-likelihood")
    return figure


def plot_trace(trace, num_draws=500):
    components = {}
    for name, values in trace.items():
        values = values.reshape(*values.shape[:2], -1)
        stride = max(1, values.shape[1] // num_draws)
        for arg in range(values.shape[2]):
            components[f"{name} {arg}"] = values[:, ::stride, arg].T
    size = (6.5, 1.2 * len(components))
    figure, _ = plot.trace_panel(components, size)
    return figure


def build_image_panels(observation, estimate, samples):
    return {
        "observation": (draw_image, observation, "observation"),
        "estimate": (draw_image, estimate, "median estimate"),
        "samples": (draw_image, samples, "posterior samples"),
    }


def build_posterior_panels(trace, prior_samples, true_values):
    samples = {name: flatten(values) for name, values in trace.items()}
    color = (samples["color"], true_values["color"], "rgb", "color")
    scale = (samples["scale"], true_values["scale"], "xyz", "scale [m]")
    panels = {
        "shift": (draw_shift, samples["shift"], true_values["shift"]),
        "color": (draw_channels, *color),
        "scale": (draw_channels, *scale),
        "theta": (draw_theta, samples["theta"][:, 0], true_values["theta"]),
        "classes": (draw_simplex, samples["classes"]),
    }
    for name in ["shininess", "ambient", "diffuse", "specular"]:
        label = COMPONENT_LABELS[name][0]
        args = (samples[name], flatten_prior(prior_samples[name]), label)
        panels[name] = (draw_scalar, *args)
    return panels


def build_prior_panels(priors, prior_samples, fitted_values):
    panels = {}
    for prior in priors:
        if prior.name == "classes":
            panels["classes"] = (draw_simplex, prior_samples["classes"])
        else:
            samples = flatten_prior(prior_samples[prior.name])
            values = np.asarray(fitted_values.get(prior.name, samples[:0]))
            values = values.reshape(-1, samples.shape[1])
            args = (prior.distribution, samples, values)
            panels.update(build_density_panels(prior.name, *args))
    return panels


def build_density_panels(name, distribution, samples, values):
    panels = {}
    for arg, label in enumerate(COMPONENT_LABELS[name]):
        grid = np.linspace(*np.percentile(samples[:, arg], [0.05, 99.95]), 400)
        densities = compute_marginal_densities(distribution, grid, arg)
        args = (grid, densities, values[:, arg], label)
        panels[f"{name}_{arg}"] = (draw_prior_density, *args)
    return panels


def build_bijector_panels(inverse_before, inverse_after):
    panels = {}
    for name, before in inverse_before.items():
        before = flatten_prior(before)
        after = flatten_prior(inverse_after[name])
        for arg, label in enumerate(COMPONENT_LABELS[name]):
            args_before = (before[:, arg], label, "before")
            args_after = (after[:, arg], label, "after")
            panels[f"{name}_{arg}_before"] = (draw_bijector_fit, *args_before)
            panels[f"{name}_{arg}_after"] = (draw_bijector_fit, *args_after)
    return panels


def build_mosaic(images, shape):
    images = paz.image.denormalize(jp.array(images))
    return paz.draw.mosaic(np.array(images), shape, border=4)


def draw_image(figure, cell, image, title):
    axis = figure.add_subplot(cell)
    axis.imshow(image)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def draw_shift(figure, cell, samples, true_shift):
    ratios = dict(width_ratios=[5, 1], height_ratios=[1, 5])
    grid = cell.subgridspec(2, 2, wspace=0.03, hspace=0.03, **ratios)
    axis = figure.add_subplot(grid[1, 0])
    axis_x = figure.add_subplot(grid[0, 0], sharex=axis)
    axis_y = figure.add_subplot(grid[1, 1], sharey=axis)
    x, y, mass = compute_joint_mass(*samples.T)
    levels = compute_highest_density_levels(mass)
    filled_levels = [*levels, mass.max()]
    axis.contourf(x, y, mass.T, levels=filled_levels, cmap="viridis")
    axis.contour(x, y, mass.T, levels=levels, colors="w", linewidths=0.3)
    inside = mass >= levels[0]
    axis.set_xlim(compute_limits(x[inside.any(axis=1)], true_shift[0]))
    axis.set_ylim(compute_limits(y[inside.any(axis=0)], true_shift[1]))
    # World +x points to the left of the image; flip so both agree.
    axis.invert_xaxis()
    draw_point_estimates(axis, np.median(samples, axis=0), true_shift)
    axis_x.fill_between(x, mass.sum(axis=1), color=DANDELION, alpha=0.6)
    axis_y.fill_betweenx(y, mass.sum(axis=0), color=DANDELION, alpha=0.6)
    axis_x.axis("off")
    axis_y.axis("off")
    finish_axis(axis, *COMPONENT_LABELS["shift"])


def compute_joint_mass(x, y, num_bins=120):
    # A histogram keeps stray chains from widening the density of the mode.
    edges_x = compute_central_edges(x, num_bins)
    edges_y = compute_central_edges(y, num_bins)
    counts, _, _ = np.histogram2d(x, y, [edges_x, edges_y])
    mass = ndimage.gaussian_filter(counts / len(x), 2.0)
    return compute_centers(edges_x), compute_centers(edges_y), mass


def compute_central_edges(samples, num_bins):
    low, high = np.percentile(samples, [10, 90])
    padding = 1.5 * (high - low)
    return np.linspace(low - padding, high + padding, num_bins + 1)


def compute_centers(edges):
    return (edges[:-1] + edges[1:]) / 2


def compute_highest_density_levels(mass):
    masses = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    sorted_mass = np.sort(mass.ravel())[::-1]
    args = np.searchsorted(np.cumsum(sorted_mass), masses)
    return sorted_mass[np.minimum(args, mass.size - 1)]


def compute_limits(values, true_value, padding=0.6):
    low = min(values.min(), true_value)
    high = max(values.max(), true_value)
    margin = padding * (high - low)
    return low - margin, high + margin


def draw_point_estimates(axis, median, true_point):
    line = dict(color=DANDELION, linestyle="--", linewidth=0.7)
    axis.axvline(median[0], **line)
    axis.axhline(median[1], **line)
    edge = dict(markeredgecolor="k", markeredgewidth=0.4, linestyle="none")
    axis.plot(*median, "s", color=DANDELION, label="median", **edge)
    axis.plot(*true_point, "*", color="tab:red", label="true", **edge)
    axis.legend(handlelength=1.2, loc="best")


def draw_channels(figure, cell, samples, true_values, names, label):
    axis = figure.add_subplot(cell)
    colors = ["tab:red", "tab:green", "tab:blue"]
    channels = zip(samples.T, true_values, names, colors)
    for values, true_value, name, color in channels:
        draw_density(axis, values, color, name)
        draw_true_value(axis, true_value, color)
    draw_true_value(axis, np.nan, "w").set_label("true")
    finish_density_axis(axis, label)


def draw_true_value(axis, value, color):
    style = dict(color=color, markeredgecolor="k", markeredgewidth=0.5)
    style.update(markersize=4, linestyle="none", clip_on=False, zorder=5)
    return axis.plot(value, 0, "o", **style)[0]


def draw_scalar(figure, cell, samples, prior_samples, label):
    axis = figure.add_subplot(cell)
    draw_density(axis, samples[:, 0], DANDELION, "posterior")
    limits = axis.get_xlim()
    grid = np.linspace(*limits, 300)
    density = stats.gaussian_kde(thin(prior_samples[:, 0]))(grid)
    axis.plot(grid, density, color=GRAY, linestyle="--", label="prior")
    axis.set_xlim(limits)
    finish_density_axis(axis, label)


def draw_theta(figure, cell, samples, true_theta):
    axis = figure.add_subplot(cell, projection="polar")
    limits = (-np.pi, np.pi)
    values = (samples + np.pi) % (2 * np.pi) - np.pi
    counts, edges = np.histogram(values, bins=90, range=limits)
    radius = np.sqrt(counts / counts.max())
    bars = dict(width=np.diff(edges), align="edge", color=DANDELION)
    axis.bar(edges[:-1], radius, **bars)
    axis.plot([true_theta, true_theta], [0, 1], color="tab:red", label="true")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_yticks([])
    axis.set_xticks(np.pi / 2 * np.arange(4))
    axis.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$-\pi/2$"])
    axis.set_xlabel(COMPONENT_LABELS["theta"][0])


def draw_simplex(figure, cell, probabilities):
    axis = figure.add_subplot(cell)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.75**0.5]])
    x, y = (probabilities @ corners).T
    hexagons = dict(gridsize=24, extent=(0, 1, 0, 1), bins="log", mincnt=1)
    bins = axis.hexbin(x, y, cmap="viridis", linewidths=0, **hexagons)
    # Bins without samples take the lowest colour instead of leaving holes.
    empty_color = plt.get_cmap("viridis")(0.0)
    axis.add_patch(plt.Polygon(corners, color=empty_color, zorder=0))
    triangle = plt.Polygon(corners, fill=False, edgecolor="k", linewidth=0.6)
    axis.add_patch(triangle)
    bins.set_clip_path(triangle)
    draw_simplex_labels(axis, corners, probabilities.mean(axis=0))
    axis.set_xlim(-0.2, 1.2)
    axis.set_ylim(-0.25, 1.05)
    axis.set_aspect("equal")
    axis.axis("off")
    colorbar_axis = axis.inset_axes([1.0, 0.2, 0.04, 0.6])
    figure.colorbar(bins, cax=colorbar_axis, label="samples")


def draw_simplex_labels(axis, corners, mean_probabilities):
    offsets = [(0, -4), (0, -4), (0, 4)]
    alignments = ["top", "top", "bottom"]
    names = ["sphere", "cube", "cylinder"]
    labels = zip(corners, names, mean_probabilities, offsets, alignments)
    for corner, name, probability, offset, alignment in labels:
        text = f"{name}\n{probability:.2f}"
        style = dict(ha="center", va=alignment, fontsize=7)
        axis.annotate(text, corner, offset, textcoords="offset points", **style)


def draw_prior_density(figure, cell, grid, densities, values, label):
    axis = figure.add_subplot(cell)
    total = densities.sum(axis=0)
    axis.plot(grid, densities.T, color=DANDELION, linewidth=0.6, linestyle=":")
    axis.plot(grid, total, color=DANDELION)
    axis.fill_between(grid, total, color=DANDELION, alpha=0.3, linewidth=0)
    rug = dict(color="k", markersize=5, markeredgewidth=0.4, clip_on=False)
    # An empty unclipped rug would break the constrained layout.
    axis.plot(values, np.zeros(len(values)), "|", in_layout=False, **rug)
    finish_density_axis(axis, label)


def draw_bijector_fit(figure, cell, inverse_samples, label, title):
    axis = figure.add_subplot(cell)
    grid = np.linspace(-4, 4, 300)
    normal = stats.norm.pdf(grid)
    axis.plot(grid, normal, color=GRAY, linestyle="--", label="normal")
    axis.fill_between(grid, normal, color=GRAY, alpha=0.2, linewidth=0)
    draw_density(axis, inverse_samples, "tab:blue", "inverse")
    axis.set_title(title, fontsize=8)
    finish_density_axis(axis, label)


def draw_density(axis, samples, color, label):
    grid, density = compute_density(samples)
    axis.plot(grid, density, color=color, label=label)
    axis.fill_between(grid, density, color=color, alpha=0.3, linewidth=0)


def finish_density_axis(axis, x_label):
    axis.set_ylim(bottom=0)
    finish_axis(axis, x_label, "density")


def finish_axis(axis, x_label, y_label):
    plot.clean(axis, "box")
    plot.set_labels(axis, x=x_label, y=y_label, labelpad=4)
    handles = axis.get_legend_handles_labels()[0]
    if handles:
        bottom, top = axis.get_ylim()
        axis.set_ylim(top=top + 0.35 * (top - bottom))
        kwargs = dict(handlelength=1.0, loc="upper left", ncols=len(handles))
        legend = axis.legend(columnspacing=0.8, borderaxespad=0.2, **kwargs)
        legend.set_in_layout(False)


def compute_marginal_densities(distribution, grid, arg):
    if isinstance(distribution, paz.distributions.MixtureSameFamily):
        components = distribution.components_distribution
        weights = np.asarray(distribution.mixture_distribution.probs)
        means = np.asarray(components.loc)[:, arg]
        stdvs = np.asarray(components.scale_diag)[:, arg]
        densities = stats.norm.pdf(grid[:, None], means, stdvs) * weights
    else:
        densities = np.asarray(distribution.prob(grid[:, None]))[:, [arg]]
    return densities.T


def compute_density(samples):
    samples = thin(samples)
    grid = compute_grid(samples, 300)
    return grid, stats.gaussian_kde(samples)(grid)


def compute_grid(samples, num_points):
    low, high = np.percentile(samples, [0.5, 99.5])
    padding = 0.25 * (high - low)
    return np.linspace(low - padding, high + padding, num_points)


def thin(samples, num_samples=20_000):
    stride = max(1, len(samples) // num_samples)
    return samples[::stride]


def flatten(values):
    values = np.asarray(values)
    return values.reshape(values.shape[0] * values.shape[1], -1)


def flatten_prior(values):
    values = np.asarray(values)
    return values.reshape(len(values), -1)
