from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import paz
import paz.graphics.renderer as paz_renderer
import paz.utils.plot as plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot.configure()
H, W = 2**8, 2**8
IMAGE_SHAPE = (H, W)
Y_FOV = jp.pi / 4.0
TILES = (4, 4)
CHUNK_SIZE = 2**10
FINITE_EPSILON = 1e-3
OUTLIER_RATIO = 1.25
CURVATURE_RATIO = 1e-3
LUMINANCE_WEIGHTS = jp.array([0.2126, 0.7152, 0.0722])
SIGNED_CMAP = "RdBu_r"
SOFT_OCCLUSION = paz_renderer.compute_soft_occlusion
CAMERA_ARGS = (
    jp.array([0.0, 2.0, 2.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)
CAMERA_POSE = paz.SE3.view_transform(*CAMERA_ARGS)
PLANE_POSE = paz.SE3.translation(jp.array([0.0, -1.0, 0.0]))
PLANE_MATERIAL = paz.graphics.Material(jp.ones(3), 0.3, 0.1, 1.0, 1.0, 0.3)
SHAPE_MATERIAL = paz.graphics.Material(
    jp.array([0.8, 0.1, 0.2]), 0.3, 0.7, 0.5, 10.0
)


def hide_ticks(axis):
    axis.set_xticks([])
    axis.set_yticks([])


def build_cmap(transition_steps=64):
    ylgn = plt.colormaps["YlGn"]
    transition = ("w2y_transition", ["white", ylgn(0.0)])
    transition = mcolors.LinearSegmentedColormap.from_list(*transition)
    transition = transition(np.linspace(0, 1, transition_steps))
    colors = np.vstack((transition[:-1], ylgn(np.linspace(0, 1, 512))))
    return mcolors.LinearSegmentedColormap.from_list("SmoothWhiteYlGn", colors)


def build_area_light():
    intensity = jp.full(3, 0.9)
    corner = jp.array([8.0, 18.0, -2.0])
    edge1 = jp.array([4.0, 0.0, 0.0])
    edge2 = jp.array([0.0, 0.0, 4.0])
    args = intensity, corner, edge1, edge2, 10, 10, jax.random.key(0)
    return paz.graphics.AreaLight(*args)


def build_render(shape):
    args = shape, Y_FOV, CAMERA_POSE
    kwargs = dict(lights=build_area_light(), mask=None, tiles=TILES)
    kwargs.update(chunk_size=CHUNK_SIZE, shadows=True, num_bounces=4)
    return paz.partial(paz.graphics.render, *args, **kwargs)


def build_plane():
    return paz.graphics.Plane(PLANE_POSE, PLANE_MATERIAL)


def build_scene(shape_transform=jp.eye(4)):
    sphere = paz.graphics.Sphere(shape_transform, SHAPE_MATERIAL)
    return paz.graphics.Scene([sphere, build_plane()])


def configure_soft_occlusion():
    paz_renderer.compute_soft_occlusion = paz.partial(SOFT_OCCLUSION, slope=1.0)


def compute_autodiff_gradient(function, args, basis):
    _, gradient = jax.jvp(function, (args,), (basis,))
    return gradient


def compute_finite_difference_gradients(function, args, basis):
    base = function(args)
    high = function(args + FINITE_EPSILON * basis)
    low = function(args - FINITE_EPSILON * basis)
    central = (high - low) / (2.0 * FINITE_EPSILON)
    forward = (high - base) / FINITE_EPSILON
    backward = (base - low) / FINITE_EPSILON
    return central, forward, backward


def compute_luminance_map(gradients):
    args = gradients, LUMINANCE_WEIGHTS
    return jp.tensordot(*args, axes=([-1], [0]))


def select_stable_values(image, stable_mask):
    array = np.asarray(image)
    if stable_mask is None:
        return array
    return array[np.asarray(stable_mask)]


def compute_positive_limits(*images):
    image_max = 0.0
    image_min = np.inf
    for image in images:
        array = np.asarray(image)
        if array.size == 0:
            continue
        image_max = max(image_max, float(array.max()))
        positive = array[array > 0.0]
        if len(positive) > 0:
            image_min = min(image_min, float(positive.min()))
    if not np.isfinite(image_min):
        image_min = 1e-8
    return image_min, max(image_max, image_min)


def compute_signed_limit(*images):
    value = 0.0
    for image in images:
        array = np.asarray(image)
        if array.size == 0:
            continue
        value = max(value, float(np.abs(array).max()))
    return max(value, 1e-8)


def show_colorbar(axis, image, label):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="7%", pad=0.1)
    colorbar = axis.figure.colorbar(image, cax=cax)
    colorbar.ax.set_ylabel(label, rotation=-90, va="bottom")


def show_scalar_image(image, cmap, norm, label, title):
    figure, axis = plt.subplots(figsize=(5, 5))
    artist = axis.imshow(image, cmap=cmap, norm=norm)
    axis.set_title(title)
    hide_ticks(axis)
    plot.hide_spines(axis, "all")
    show_colorbar(axis, artist, label)
    return figure


def show_magnitude_image(gradients, title, stable_mask=None):
    magnitude = jp.linalg.norm(gradients, axis=-1)
    values = select_stable_values(magnitude, stable_mask)
    v_min, v_max = compute_positive_limits(values)
    norm = mcolors.LogNorm(vmin=v_min, vmax=v_max)
    return show_scalar_image(
        magnitude, build_cmap(), norm, "L2 Gradient Norm", title
    )


def show_signed_image(gradients, title, stable_mask=None):
    signed = compute_luminance_map(gradients)
    values = select_stable_values(signed, stable_mask)
    limit = compute_signed_limit(values)
    norm = mcolors.Normalize(vmin=-limit, vmax=limit)
    return show_scalar_image(
        signed, SIGNED_CMAP, norm, "Signed Sensitivity", title
    )


def show_magnitude_comparison(autodiff, finite, stable_mask):
    magnitude_auto = jp.linalg.norm(autodiff, axis=-1)
    magnitude_finite = jp.linalg.norm(finite, axis=-1)
    magnitude_diff = jp.abs(magnitude_auto - magnitude_finite)
    auto_values = select_stable_values(magnitude_auto, stable_mask)
    finite_values = select_stable_values(magnitude_finite, stable_mask)
    diff_values = select_stable_values(magnitude_diff, stable_mask)
    v_min, v_max = compute_positive_limits(auto_values, finite_values)
    diff_min, diff_max = compute_positive_limits(diff_values)
    norm = mcolors.LogNorm(vmin=v_min, vmax=v_max)
    diff_norm = mcolors.LogNorm(vmin=diff_min, vmax=diff_max)
    titles = ["Autodiff", "Finite Difference", "Absolute Difference"]
    images = [magnitude_auto, magnitude_finite, magnitude_diff]
    norms = [norm, norm, diff_norm]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    for axis, image, title, image_norm in zip(axes[0], images, titles, norms):
        artist = axis.imshow(image, cmap=build_cmap(), norm=image_norm)
        axis.set_title(title)
        hide_ticks(axis)
        plot.hide_spines(axis, "all")
        show_colorbar(axis, artist, "L2 Gradient Norm")
    return figure


def show_signed_comparison(autodiff, finite, stable_mask):
    signed_auto = compute_luminance_map(autodiff)
    signed_finite = compute_luminance_map(finite)
    signed_diff = signed_auto - signed_finite
    auto_values = select_stable_values(signed_auto, stable_mask)
    finite_values = select_stable_values(signed_finite, stable_mask)
    diff_values = select_stable_values(signed_diff, stable_mask)
    limit = compute_signed_limit(auto_values, finite_values)
    diff_limit = compute_signed_limit(diff_values)
    norm = mcolors.Normalize(vmin=-limit, vmax=limit)
    diff_norm = mcolors.Normalize(vmin=-diff_limit, vmax=diff_limit)
    titles = ["Autodiff", "Finite Difference", "Difference"]
    images = [signed_auto, signed_finite, signed_diff]
    norms = [norm, norm, diff_norm]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    for axis, image, title, image_norm in zip(axes[0], images, titles, norms):
        artist = axis.imshow(image, cmap=SIGNED_CMAP, norm=image_norm)
        axis.set_title(title)
        hide_ticks(axis)
        plot.hide_spines(axis, "all")
        show_colorbar(axis, artist, "Signed Sensitivity")
    return figure


def show_unstable_pixels(stable_mask):
    unstable = jp.logical_not(stable_mask).astype(float)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    return show_scalar_image(
        unstable, "Reds", norm, "Unstable Pixel", "Unstable FD Pixels"
    )


def compute_stable_mask(autodiff, finite, forward, backward):
    autodiff_mag = jp.linalg.norm(autodiff, axis=-1)
    finite_mag = jp.linalg.norm(finite, axis=-1)
    curvature = jp.linalg.norm(forward - backward, axis=-1)
    ratio = finite_mag / jp.maximum(autodiff_mag, 1.0)
    scale = jp.maximum(jp.maximum(autodiff_mag, finite_mag), 1.0)
    is_local = curvature < CURVATURE_RATIO * scale
    is_bounded = ratio < OUTLIER_RATIO
    is_finite = jp.logical_and(jp.isfinite(ratio), jp.isfinite(curvature))
    return jp.logical_and(jp.logical_and(is_local, is_bounded), is_finite)


def compute_gradient_metrics(autodiff, finite, stable_mask=None):
    if stable_mask is not None:
        autodiff = autodiff[stable_mask]
        finite = finite[stable_mask]
    autodiff = jp.ravel(autodiff)
    finite = jp.ravel(finite)
    difference = autodiff - finite
    autodiff_norm = jp.linalg.norm(autodiff)
    finite_norm = jp.linalg.norm(finite)
    diff_norm = jp.linalg.norm(difference)
    product = jp.sum(autodiff * finite)
    cosine = product / (autodiff_norm * finite_norm + 1e-12)
    relative = diff_norm / (finite_norm + 1e-12)
    return autodiff_norm, finite_norm, diff_norm, relative, cosine


def print_metric_block(name, metrics):
    autodiff_norm, finite_norm, diff_norm, relative, cosine = metrics
    print(f"  {name} AD norm: {float(autodiff_norm):.6f}")
    print(f"  {name} FD norm: {float(finite_norm):.6f}")
    print(f"  {name} diff norm: {float(diff_norm):.6f}")
    print(f"  {name} relative error: {float(relative):.6f}")
    print(f"  {name} cosine similarity: {float(cosine):.6f}")


def print_gradient_comparison(axis_arg, autodiff, finite, stable_mask):
    stable_pixels = int(jp.sum(stable_mask))
    total_pixels = stable_mask.size
    raw_metrics = compute_gradient_metrics(autodiff, finite)
    stable_metrics = compute_gradient_metrics(autodiff, finite, stable_mask)
    print(f"Axis {axis_arg} gradient comparison:")
    print(f"  stable pixels: {stable_pixels}/{total_pixels}")
    print_metric_block("stable", stable_metrics)
    print_metric_block("raw", raw_metrics)


def save(figure, filepath):
    figure.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(figure)


def save_axis_images(axis_arg, autodiff, finite, stable_mask):
    save(
        show_magnitude_image(autodiff, "Autodiff Magnitude"),
        f"{axis_arg}-axis_translation_autodiff_magnitude.pdf",
    )
    save(
        show_magnitude_image(
            finite, "Finite Difference Magnitude", stable_mask
        ),
        f"{axis_arg}-axis_translation_finite_difference_magnitude.pdf",
    )
    save(
        show_signed_image(autodiff, "Autodiff Signed"),
        f"{axis_arg}-axis_translation_autodiff_signed.pdf",
    )
    save(
        show_signed_image(finite, "Finite Difference Signed", stable_mask),
        f"{axis_arg}-axis_translation_finite_difference_signed.pdf",
    )
    save(
        show_magnitude_comparison(autodiff, finite, stable_mask),
        f"{axis_arg}-axis_translation_magnitude_comparison.pdf",
    )
    save(
        show_signed_comparison(autodiff, finite, stable_mask),
        f"{axis_arg}-axis_translation_signed_comparison.pdf",
    )
    save(
        show_unstable_pixels(stable_mask),
        f"{axis_arg}-axis_translation_unstable_finite_pixels.pdf",
    )


def build_renderer(render):
    def render_shape(transform_args):
        transform = paz.SE3.translation(transform_args)
        image, _ = render(scene=build_scene(transform))
        return image

    return render_shape


def plot_jacobian(function, args):
    print(f"Finite-difference epsilon: {FINITE_EPSILON:.1e}")
    print(f"FD curvature ratio: {CURVATURE_RATIO:.1e}")
    print(f"FD magnitude ratio: {OUTLIER_RATIO:.2f}")
    autodiff = jax.jit(
        lambda args, basis: compute_autodiff_gradient(function, args, basis)
    )
    finite = jax.jit(
        lambda args, basis: compute_finite_difference_gradients(
            function, args, basis
        )
    )
    for axis_arg, basis in enumerate(jp.eye(len(args))):
        autodiff_gradient = jp.array(autodiff(args, basis))
        finite_gradients = finite(args, basis)
        finite_gradient, forward, backward = map(jp.array, finite_gradients)
        stable_mask = compute_stable_mask(
            autodiff_gradient, finite_gradient, forward, backward
        )
        image_args = axis_arg, autodiff_gradient, finite_gradient, stable_mask
        print_gradient_comparison(*image_args)
        save_axis_images(*image_args)


def main():
    configure_soft_occlusion()
    render = build_render(IMAGE_SHAPE)
    image, _ = render(scene=build_scene())
    image = paz.image.resize(image, (H // 2, W // 2), "bilinear")
    image = paz.image.denormalize(image)
    paz.image.write("color_gradients_scene.png", image)
    plot_jacobian(build_renderer(render), jp.zeros(3))


if __name__ == "__main__":
    main()
