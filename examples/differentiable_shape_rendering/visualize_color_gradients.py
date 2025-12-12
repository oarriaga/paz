# TODO we should do a visualization in which we show multiple materials params
import jax

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jp
import numpy as np
import jax
import paz

image_filename = "color_scene.png"
H, W = 128, 128
H, W = 1028, 1028

resize_factor = 1
resized_image_shape = (H // resize_factor, W // resize_factor)
image_shape = (H, W)
y_FOV = jp.pi / 4.0
min_color_gradient = 1e-4
max_color_gradient = 10
# axes = paz.graphics.load("axes.json")

# min_depth_gradient = 1e-2
min_depth_gradient = -1e1
max_depth_gradient = 1e1
camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 1.625, 1.625]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)

# camera_pose = paz.SE3.view_transform(
#     jp.array([0.0, 0.0, 3.5]),
#     jp.array([0.0, 0.0, 0.0]),
#     jp.array([0.0, 1.0, 0.0]),
# )


plane_material = paz.graphics.Material(
    jp.array([0.8, 0.8, 0.8]), 0.3, 0.1, 1.0, 1.0, 0.5
)
shape_material = paz.graphics.Material(
    jp.array([0.8, 0.1, 0.2]), 0.3, 0.7, 0.5, 10.0
)
plane_pose = paz.SE3.translation(jp.array([0.0, -1.0, 0.0]))
plane_angle = paz.SE3.rotation_x(-jp.pi / 2)
# plane = paz.graphics.Plane(plane_pose @ plane_angle, material=plane_material)
shape_pose = paz.SE3.translation(jp.array([0.1, 0.0, -0.25]))
plane = paz.graphics.Plane(plane_pose, material=plane_material)
shape = paz.graphics.Sphere(shape_pose, shape_material)
scene = paz.graphics.Scene([shape, plane])
rays = paz.graphics.camera.build_rays(image_shape, y_FOV, camera_pose)
light_position = jp.array([8.0, 18.0, -2.0])
lights = [
    paz.graphics.PointLight(jp.full(3, 0.9), light_position),
    # paz.graphics.PointLight(jp.full(3, 0.9), jp.array([2.0, 2.0, 3.0])),
    # paz.graphics.PointLight(jp.full(3, 0.9), jp.array([2.2, 1.3, 1.2])),
]


def build_renderer(SE3_transform, render_arg=0):
    def _build_renderer(transform_args):
        transform = SE3_transform(transform_args)
        shape = paz.graphics.Sphere(transform @ shape_pose, shape_material)
        # shape = paz.graphics.Sphere(shape_pose, shape_material)
        # new_scene = paz.graphics.Scene([shape, plane] + axes.nodes)
        new_scene = paz.graphics.Scene([shape, plane])
        # new_light_position = paz.algebra.transform(transform, light_position)
        # new_light_position = new_light_position + light_position
        # lights = [paz.graphics.PointLight(jp.full(3, 0.9), new_light_position)]
        values = paz.graphics.render(
            image_shape,
            camera_pose,
            rays,
            new_scene,
            lights,
            mask=None,
            shadows=False,
        )

        x = values[render_arg]
        if render_arg == 0:
            x = paz.image.resize(
                x, (H // resize_factor, W // resize_factor), "bilinear"
            )
        return x

    return _build_renderer


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
    plt.savefig(image_filename, bbox_inches="tight", pad_inches=0.0)
    print("Jacobian shape:", jacobian.shape)
    num_channels = jacobian.shape[-1]
    for channel_arg in range(num_channels):
        gradients = jacobian[..., channel_arg]
        visualize_gradients(jp.array(gradients), f"{channel_arg}_{filename}")


def create_white_ylgn_cmap(transition_steps=64):
    ylgn_cmap = plt.colormaps["YlGn"]
    start_yellow = ylgn_cmap(0.0)
    colors = ["white", start_yellow]
    cmap = mcolors.LinearSegmentedColormap.from_list("w2y_transition", colors)
    transition_colors = cmap(np.linspace(0, 1, transition_steps))
    ylgn_colors = ylgn_cmap(np.linspace(0, 1, 512))
    colors = np.vstack((transition_colors[:-1], ylgn_colors))
    return mcolors.LinearSegmentedColormap.from_list("SmoothWhiteYlGn", colors)


def visualize_gradients(
    gradients,
    filename=None,
    mode="magnitude",
    # cmap_channels=create_white_ylgn_cmap(),
    cmap_channels="PiYG",
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


def visualize_gradient_magnitude(gradients, cmap):
    config = paz.plot.build_configuration(figsize=(1080, 1080))
    figure, axis = plt.subplots(figsize=config.figsize)
    paz.plot.hide_all_axes(axis)
    magnitude = jp.linalg.norm(gradients, axis=-1)
    image = plt.imshow(
        magnitude,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=min_color_gradient, vmax=max_color_gradient),
    )
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.0)
    colorbar = axis.figure.colorbar(image, cax=cax)
    colorbar.ax.set_ylabel(
        "L2 Gradient Norm", rotation=-90, va="bottom", labelpad=10, fontsize=32
    )
    colorbar.ax.tick_params(labelsize=32)
    axis.set_xticks([])
    axis.set_yticks([])
    return figure


def visualize_gradient_channels(gradients, cmap):
    config = paz.plot.build_configuration(figsize=(1080, 1080))
    figure, axis = plt.subplots(figsize=config.figsize)
    paz.plot.hide_all_axes(axis)
    magnitude = gradients
    # magnitude = jp.abs(gradients)
    print(magnitude.max(), magnitude.min())
    image = plt.imshow(
        magnitude,
        cmap=cmap,
        norm=mcolors.SymLogNorm(
            linthresh=0.01, vmin=min_depth_gradient, vmax=max_depth_gradient
        ),
    )
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="7%", pad=0.05)
    colorbar = axis.figure.colorbar(image, cax=cax)
    colorbar.ax.set_ylabel("Gradient", rotation=-90, va="bottom", labelpad=10)
    axis.set_xticks([])
    axis.set_yticks([])
    return figure


render_with_constraints = build_renderer(paz.SE3.translation, 0)
plot_jacobian(
    render_with_constraints,
    jp.array([0.0, 0.0, 0.0]),
    "image_translation_gradients.pdf",
)
