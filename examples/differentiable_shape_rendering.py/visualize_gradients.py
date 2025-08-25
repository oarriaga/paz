import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import paz
from paz import SE3
import numpy as np
from paz.graphics import SPHERE

# from paz.graphics import CUBE
from paz.graphics import PointLight

import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_white_ylgn_cmap(transition_steps=64):
    ylgn_cmap = plt.colormaps["YlGn"]
    start_yellow = ylgn_cmap(0.0)
    transition_cmap = mcolors.LinearSegmentedColormap.from_list(
        "w2y_transition", ["white", start_yellow]
    )
    transition_colors = transition_cmap(np.linspace(0, 1, transition_steps))
    ylgn_colors = ylgn_cmap(np.linspace(0, 1, 512))
    colors = np.vstack((transition_colors[:-1], ylgn_colors))
    return mcolors.LinearSegmentedColormap.from_list("SmoothWhiteYlGn", colors)


material = paz.graphics.Material(jp.array([1.0, 0.2, 0.1]), 0.4, 0.4, 0.8, 50)
# light = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([5.0, 5.0, -5.0]))]
light = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([0.0, 0.0, -5.0]))]
camera_transform = SE3.view_transform(
    jp.array([0.0, 0.0, -2.05]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

W = 2040
H = 2040
y_FOV = jp.pi / 3
key = jax.random.PRNGKey(777)
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_transform)


def visualize_gradient_magnitude(gradients, cmap):
    config = paz.plot.build_configuration(figsize=(1080, 1080))
    figure, axis = plt.subplots(figsize=config.figsize)
    paz.plot.hide_all_axes(axis)
    magnitude = jp.linalg.norm(gradients, axis=-1)
    print(magnitude.min(), magnitude.max())
    image = plt.imshow(
        magnitude,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=0.1, vmax=10),
    )
    # plt.colorbar(image, label="L2 Norm of Gradient Vector")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="7%", pad=0.1)
    colorbar = axis.figure.colorbar(image, cax=cax)
    colorbar.ax.set_ylabel("L2 Gradient Norm", rotation=-90, va="bottom")
    axis.set_xticks([])
    axis.set_yticks([])
    return figure


def visualize_gradient_channels(gradients, cmap):
    config = paz.plot.build_configuration(figsize=(1080, 1080))
    num_channels = gradients.shape[-1]
    # figsize = (6 * num_channels, 5) if num_channels > 1 else (7, 6)
    figure, axes = plt.subplots(
        1, num_channels, figsize=config.figsize, squeeze=False
    )
    axes = axes.flatten()
    for channel_arg, axis in enumerate(axes):
        channel_data = gradients[..., channel_arg]
        v_max = np.abs(channel_data).max()
        image = axis.imshow(channel_data, cmap=cmap, vmin=-v_max, vmax=v_max)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.1)
    return figure


def visualize_gradients(
    # gradients, mode="magnitude", cmap_channels="RdBu_r", cmap_magnitude="magma"
    gradients,
    filename=None,
    mode="magnitude",
    # cmap_channels="YlGn",
    # cmap_magnitude="YlGn",
    cmap_channels=create_white_ylgn_cmap(),
    cmap_magnitude=create_white_ylgn_cmap(),
):
    if gradients.ndim not in [2, 3]:
        raise ValueError("'gradients' must have 2 or 3 dimensions")
    if mode not in ["channels", "magnitude"]:
        raise ValueError("Mode must be either 'channels' or 'magnitude'.")
    if gradients.ndim == 2:
        if mode == "magnitude":
            print("'magnitude' is for multi-channel data, switch to 'channels'")
        mode = "channels"
        gradients = gradients[..., jp.newaxis]

    if mode == "magnitude":
        figure = visualize_gradient_magnitude(gradients, cmap_magnitude)
    elif mode == "channels":
        figure = visualize_gradient_channels(gradients, cmap_channels)
    paz.plot.write_or_show(figure, filename)
    plt.show()


def plot_jacobian(function, args, filename=None):
    compute_jacobian = jax.jit(jax.jacfwd(function))
    jacobian = compute_jacobian(args)
    config = paz.plot.build_configuration(figsize=(1080, 1080))
    figure, axis = plt.subplots(figsize=config.figsize)
    image = function(args)
    axis.imshow(image)
    paz.plot.hide_all_axes(axis)
    axis.set_xticks([])
    axis.set_yticks([])
    paz.plot.write_or_show(figure, filename)
    num_channels = jacobian.shape[-1]
    for channel_arg in range(num_channels):
        gradients = jacobian[..., channel_arg]
        visualize_gradients(jp.array(gradients), f"{channel_arg}_{filename}")


def build_renderer(shape_type, SE3_transform, render_arg=1):
    pattern = paz.graphics.Pattern(
        jp.eye(4), paz.graphics.NO_PATTERN, jp.ones((1, 1, 3))
    )

    def _build_renderer(transform_args):
        transform = SE3_transform(transform_args)
        shape = paz.graphics.Shape(transform, shape_type, pattern, material)
        render = paz.graphics.Render((H, W), jp.eye(4), rays, True)
        shape = paz.graphics.shapes.expand(shape)
        values = render(shape, jp.array([1]), light)
        x = values[render_arg]
        if render_arg == 0:
            x = paz.image.resize(x, (H // 2, W // 2), "bilinear")
        return x

    return _build_renderer


# render_with_constraints = build_renderer(CUBE, SE3.translation, 0)
# plot_jacobian(render_with_constraints, jp.array([0.0, 0.0, 0.0]))

render_with_constraints = build_renderer(SPHERE, SE3.translation, 0)
plot_jacobian(
    render_with_constraints,
    jp.array([0.0, 0.0, 0.0]),
    "sphere_translation_image_gradients.pdf",
)

render_with_constraints = build_renderer(SPHERE, SE3.scaling, 0)
plot_jacobian(
    render_with_constraints,
    jp.array([1.0, 1.0, 1.0]),
    "sphere_scaling_image_gradients.pdf",
)

# render_with_constraints = build_renderer(SPHERE, SE3.translation, 1)
# plot_jacobian(render_with_constraints, jp.array([0.0, 0.0, 0.0]))
# render_with_constraints = build_renderer(SPHERE, SE3.scaling, 1)
# plot_jacobian(render_with_constraints, jp.array([1.0, 1.0, 1.0]))
