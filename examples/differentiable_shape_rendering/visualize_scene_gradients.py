import jax
import jax.numpy as jp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import paz
import paz.utils.plot as plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot.configure()
H, W = 1028 // 2, 1028 // 2
image_shape = (H, W)
y_FOV = jp.pi / 4.0
camera_args = (
    jp.array([0.0, 2.0, 2.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)
camera_pose = paz.SE3.view_transform(*camera_args)
plane_material_args = (jp.ones(3), 0.3, 0.1, 1.0, 1.0)
shape_material_args = (jp.array([0.8, 0.1, 0.2]), 0.3, 0.7, 0.5, 10.0)
plane_material = paz.graphics.Material(*plane_material_args)
shape_material = paz.graphics.Material(*shape_material_args)
plane_pose = paz.SE3.translation(jp.array([0.0, -1.0, 0.0]))
plane = paz.graphics.Plane(plane_pose, plane_material)
scene_nodes = [paz.graphics.Sphere(material=shape_material), plane]
scene = paz.graphics.Scene(scene_nodes)
lights = [paz.graphics.PointLight(jp.full(3, 0.9), jp.array([10.0, 18.0, 0.0]))]
render_args = image_shape, y_FOV, camera_pose
render_kargs = dict(lights=lights, mask=None, tiles=(1, 1), chunk_size=1024)
render = paz.partial(paz.graphics.render, *render_args, **render_kargs)


def hide_ticks(axis):
    axis.set_xticks([])
    axis.set_yticks([])


def build_cmap(transition_steps=64):
    ylgn = plt.colormaps["YlGn"]
    white_to_yellow = ["white", ylgn(0.0)]
    transition = mcolors.LinearSegmentedColormap.from_list(
        "w2y_transition", white_to_yellow
    )
    transition = transition(np.linspace(0, 1, transition_steps))
    colors = np.vstack((transition[:-1], ylgn(np.linspace(0, 1, 512))))
    return mcolors.LinearSegmentedColormap.from_list("SmoothWhiteYlGn", colors)


def show_magnitude(gradients, cmap):
    figure, axis = plt.subplots()
    print(gradients.min(), gradients.max())
    magnitude = jp.linalg.norm(gradients, axis=-1)
    print(magnitude.min(), magnitude.max())
    norm = mcolors.LogNorm(vmin=0.00001, vmax=10)
    image = axis.imshow(magnitude, cmap=cmap, norm=norm)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="7%", pad=0.1)
    colorbar = axis.figure.colorbar(image, cax=cax)
    colorbar.ax.set_ylabel("L2 Gradient Norm", rotation=-90, va="bottom")
    hide_ticks(axis)
    return figure


def show_channels(gradients, cmap):
    num_channels = gradients.shape[-1]
    figure, axes = plt.subplots(1, num_channels, squeeze=False)
    for channel_arg, axis in enumerate(axes.flatten()):
        channel = gradients[..., channel_arg]
        v_max = np.abs(channel).max()
        image = axis.imshow(channel, cmap=cmap, vmin=-v_max, vmax=v_max)
        hide_ticks(axis)
        figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.1)
    return figure


def show_gradients(gradients, mode="magnitude", cmap=build_cmap()):
    if gradients.ndim not in [2, 3]:
        raise ValueError("'gradients' must have 2 or 3 dimensions")
    if mode not in ["channels", "magnitude"]:
        raise ValueError("Mode must be either 'channels' or 'magnitude'.")
    if gradients.ndim == 2:
        if mode == "magnitude":
            print("'magnitude' is for multi-channel data, switch to 'channels'")
        gradients = gradients[..., jp.newaxis]
        mode = "channels"
    show = show_magnitude if mode == "magnitude" else show_channels
    show(gradients, cmap)
    plt.show()


def build_renderer(transform_fn):
    def render_shape(transform_args):
        transform = transform_fn(transform_args)
        shape = paz.graphics.Sphere(transform, shape_material)
        new_scene = paz.graphics.Scene([shape, plane])
        image, _ = render(scene=new_scene, shadows=True)
        return image

    return render_shape


def plot_jacobian(function, args):
    jacobian = jax.jit(jax.jacfwd(function))(args)
    figure, axis = plt.subplots()
    axis.imshow(function(args))
    hide_ticks(axis)
    num_channels = jacobian.shape[-1]
    for channel_arg in range(num_channels):
        show_gradients(jp.array(jacobian[..., channel_arg]))


image, _ = render(scene=scene, shadows=False)
image = paz.image.resize(image, (H // 2, W // 2), "bilinear")
paz.image.show(paz.image.denormalize(image))
plot_jacobian(build_renderer(paz.SE3.translation), jp.zeros(3))
