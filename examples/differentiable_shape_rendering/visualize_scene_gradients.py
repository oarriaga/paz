# TODO make paz cache function
# TODO we should do a visualization in which we show multiple materials params
# TODO explain why shape transformations change the frame of reference
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
import paz.utils.plot as plot

H, W = 1028 // 2, 1028 // 2
image_shape = (H, W)
y_FOV = jp.pi / 4.0
camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 2.0, 2.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)


plane_material = paz.graphics.Material(jp.ones(3), 0.3, 0.1, 1.0, 1.0)
shape_material = paz.graphics.Material(
    jp.array([0.8, 0.1, 0.2]), 0.3, 0.7, 0.5, 10.0
)

# shape_size = 2.25
shape_size = 1.0
plane_pose = paz.SE3.translation(jp.array([0.0, -shape_size, 0.0]))
plane = paz.graphics.Plane(plane_pose, material=plane_material)
shape_pose = jp.eye(4)
shape = paz.graphics.Sphere(shape_pose, shape_material)
scene = paz.graphics.Scene([shape, plane])

rays = paz.graphics.camera.build_rays(image_shape, y_FOV, camera_pose)

lights = [
    paz.graphics.PointLight(jp.full(3, 0.9), jp.array([10.0, 18.0, 0.0])),
]


image, depth = paz.graphics.render(
    image_shape, camera_pose, rays, scene, lights, mask=None, shadows=False
)
image = paz.image.resize(image, (H // 2, W // 2), "bilinear")
image = paz.image.denormalize(image)
paz.image.show(image)


def build_renderer(SE3_transform, render_arg=0):
    def _build_renderer(transform_args):
        transform = SE3_transform(transform_args)
        shape = paz.graphics.Sphere(transform @ shape_pose, shape_material)
        # shape = paz.graphics.Sphere(shape_pose, shape_material)
        new_scene = paz.graphics.Scene([shape, plane])

        # camera_position = paz.algebra.transform_vectors(
        #     transform, jp.array([[-10.0, 18.0, 0.0]])
        # )
        # new_lights = [
        #     paz.graphics.PointLight(jp.full(3, 0.9), camera_position[0])
        # ]
        # new_camera_pose = transform @ camera_pose
        # new_rays = paz.graphics.camera.build_rays(
        #     image_shape, y_FOV, new_camera_pose
        # )

        values = paz.graphics.render(
            image_shape,
            camera_pose,
            rays,
            new_scene,
            lights,
            mask=None,
            shadows=True,
        )

        x = values[render_arg]
        # if render_arg == 0:
        #     x = paz.image.resize(x, (H // 2, W // 2), "bilinear")
        return x

    return _build_renderer


def plot_jacobian(function, args, filename=None):
    compute_jacobian = jax.jit(jax.jacfwd(function))
    # compute_jacobian = jax.jit(jax.jacrev(function))
    jacobian = compute_jacobian(args)
    config = plot.build_configuration(figsize=(1080, 1080))
    figure, axis = plt.subplots(figsize=config.figsize)
    image = function(args)
    axis.imshow(image)
    plot.hide_all_axes(axis)
    axis.set_xticks([])
    axis.set_yticks([])
    plot.write_or_show(figure, filename)
    num_channels = jacobian.shape[-1]
    for channel_arg in range(num_channels):
        gradients = jacobian[..., channel_arg]
        visualize_gradients(jp.array(gradients), f"{channel_arg}_{filename}")


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


def visualize_gradient_magnitude(gradients, cmap):
    config = plot.build_configuration(figsize=(1080, 1080))
    figure, axis = plt.subplots(figsize=config.figsize)
    plot.hide_all_axes(axis)
    print(gradients.min(), gradients.max())
    magnitude = jp.linalg.norm(gradients, axis=-1)
    print(magnitude.min(), magnitude.max())
    image = plt.imshow(
        magnitude,
        cmap=cmap,
        # norm=mcolors.LogNorm(vmin=0.01, vmax=10),
        norm=mcolors.LogNorm(vmin=0.00001, vmax=10),
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
    config = plot.build_configuration(figsize=(1080, 1080))
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
    plot.write_or_show(figure, filename)
    plt.show()


render_with_constraints = build_renderer(paz.SE3.translation, 0)
plot_jacobian(
    render_with_constraints,
    # jp.array([1.0, 1.0, 1.0]),
    jp.full(3, 0.0),
    "xyz_scene_with_shadows_translation_image_gradients.png",
)
