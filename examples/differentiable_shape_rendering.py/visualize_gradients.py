import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import paz
from paz import SE3
import numpy as np

material = paz.graphics.Material(jp.array([1.0, 0.2, 0.1]), 0.4, 0.4, 0.4, 200)
light = [
    paz.graphics.PointLight(
        jp.array([1.0, 1.0, 1.0]), jp.array([-10.0, 10.0, -10.0])
    )
]
camera_transform = SE3.view_transform(
    jp.array([0, 0.0, -5.0]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)
# camera = Camera((50, 50), jp.pi / 3, camera_transform)
# camera = Camera((300, 300), jp.pi / 3, camera_transform)
H = W = 300
ray_origins, ray_directions = paz.graphics.camera.build_rays(
    (300, 300), jp.pi / 3, camera_transform
)
# ray_origins, ray_directions = camera.build_rays()

# H, W = camera.H, camera.W


def visualize_gradients(
    gradients,
    mode="magnitude",
    title=None,
    cmap_channels="RdBu_r",
    cmap_magnitude="magma",
    figsize=None,
):
    """
    Visualizes a gradient image without clipping by normalizing the color scale.

    Args:
        gradients (np.ndarray):
            The gradient image, shape (H, W) for single-channel or
            (H, W, C) for multi-channel data.
        mode (str):
            Visualization mode. Can be 'channels' or 'magnitude'.
            - 'channels': Visualizes each channel separately with a centered,
                          diverging colormap (default).
            - 'magnitude': Visualizes the L2 norm across channels with a
                           sequential colormap. Best for multi-channel data.
        title (Optional[str]):
            An overall title for the plot.
        cmap_channels (str):
            Colormap for the 'channels' mode.
        cmap_magnitude (str):
            Colormap for the 'magnitude' mode.
        figsize (Optional[Tuple[int, int]]):
            Figure size for the plot. If None, it's set automatically.
    """
    # --- Input Validation ---
    if not isinstance(gradients, np.ndarray):
        raise TypeError("Input 'gradients' must be a NumPy array.")
    if gradients.ndim not in [2, 3]:
        raise ValueError(
            "Input 'gradients' must have 2 or 3 dimensions (H, W) or (H, W, C)."
        )
    if mode not in ["channels", "magnitude"]:
        raise ValueError("Mode must be either 'channels' or 'magnitude'.")

    # --- Data Preparation ---
    # Ensure gradients are 3D for consistent processing
    if gradients.ndim == 2:
        # If user requests 'magnitude' on 2D data, it's less informative.
        # Default to showing the single channel with a diverging map.
        if mode == "magnitude":
            print(
                "Warning: 'magnitude' mode is for multi-channel data. Visualizing single channel instead."
            )
        mode = "channels"
        gradients = gradients[..., np.newaxis]  # Reshape (H, W) to (H, W, 1)

    # --- Plotting ---
    if mode == "magnitude":
        magnitude = np.linalg.norm(gradients, axis=-1)

        plt.figure(figsize=figsize if figsize else (8, 6))
        im = plt.imshow(magnitude, cmap=cmap_magnitude)

        plt.colorbar(im, label="L2 Norm of Gradient Vector")
        plot_title = title if title else "Gradient Magnitude"
        plt.title(plot_title)
        plt.xticks([])
        plt.yticks([])

    elif mode == "channels":
        num_channels = gradients.shape[-1]

        # Adjust figsize automatically if not provided
        if figsize is None:
            figsize = (6 * num_channels, 5) if num_channels > 1 else (7, 6)

        fig, axes = plt.subplots(
            1, num_channels, figsize=figsize, squeeze=False
        )
        axes = axes.flatten()

        channel_names = ["R", "G", "B", "A"]

        for i, ax in enumerate(axes):
            channel_data = gradients[..., i]

            # Center the diverging colormap at zero by finding the max absolute value
            v_max = np.abs(channel_data).max()
            if v_max == 0:
                v_max = 1  # Avoid an empty range for all-zero gradients

            im = ax.imshow(
                channel_data, cmap=cmap_channels, vmin=-v_max, vmax=v_max
            )

            # Set subplot title
            chan_name = (
                channel_names[i] if i < len(channel_names) else f"Channel {i}"
            )
            ax.set_title(f"{chan_name} Gradient")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)

        if title:
            plt.suptitle(title, fontsize=16)

    plt.show()


def deprocess_image(x):
    visualize_gradients(np.array(x))
    # plt.hist(x.reshape(-1))
    # plt.show()
    # x = x - x.mean()
    # x = x / (x.std() + 1e-5)
    # x = x * 0.1

    # clip to [0, 1]
    # x = x + 0.5
    # x = jp.clip(x, 0, 1)
    return x


def postprocess_gradients(gradients):
    min_value = gradients.min()
    max_value = gradients.max()
    normalized_gradients = (gradients - min_value) / (max_value - min_value)
    print("MIN {} - MAX {}".format(min_value, max_value))
    # return gradients
    min_value = normalized_gradients.min()
    max_value = normalized_gradients.max()
    print("MIN {} - MAX {}".format(min_value, max_value))
    return gradients
    return normalized_gradients


def plot_jacobian(function, args):
    # compute_jacobian = jax.jit(jax.jacrev(function))
    compute_jacobian = jax.jit(jax.jacfwd(function))
    jacobian = compute_jacobian(args)
    image = function(args)
    print(jacobian.shape)
    num_channels = jacobian.shape[-1]
    for channel_arg in range(num_channels):
        gradients = jacobian[..., channel_arg]
        print("GRADIENTS_SHAPE", gradients.shape)
        normalized_gradients = deprocess_image(gradients)
        # grads_x = jp.repeat(normalized_gradients[:, :, 0:1], 3, axis=-1)
        # grads_y = jp.repeat(normalized_gradients[:, :, 1:2], 3, axis=-1)
        # grads_z = jp.repeat(normalized_gradients[:, :, 2:3], 3, axis=-1)
        images = jp.concatenate([image, normalized_gradients], axis=1)
        # images = jp.concatenate([image, grads_x, grads_y, grads_z], axis=1)
        # plt.imshow(normalized_gradients, cmap='viridis')
        plt.imshow(images)
        plt.colorbar()
        plt.show()


def build_renderer(shape_type, SE3_transform, render_arg=1):
    # pattern_type = jp.full((1,), paz.graphics.NO_PATTERN)
    # empty_image = jp.zeros((1, 1, 1, 3))
    # pattern = paz.graphics.Pattern(jp.eye(4), pattern_type, empty_image)
    pattern = paz.graphics.Pattern(
        jp.eye(4), paz.graphics.NO_PATTERN, jp.ones((1, 1, 3))
    )

    def _build_renderer(transform_args):
        transform = SE3_transform(transform_args)
        shape = paz.graphics.Shape(transform, shape_type, pattern, material)
        render = paz.graphics.Render(
            (H, W), jp.eye(4), (ray_origins, ray_directions), True
        )
        shape = paz.graphics.shapes.expand(shape)
        values = render(shape, jp.array([1]), light)
        # values = paz.graphics.render(
        #     [shape], light, ray_origins, ray_directions, H, W
        # )
        return values[render_arg]

    return _build_renderer


# render_with_constraints = build_renderer(paz.graphics.CUBE, SE3.translation, 1)
# plot_jacobian(render_with_constraints, jp.array([0.0, 0.0, 0.0]))

# render_with_constraints = build_renderer(paz.graphics.PLANE, SE3.translation, 1)
# plot_jacobian(render_with_constraints, jp.array([0.0, 0.0, 0.0]))

render_with_constraints = build_renderer(
    paz.graphics.SPHERE, SE3.translation, 0
)
plot_jacobian(render_with_constraints, jp.array([0.0, 0.0, 0.0]))

render_with_constraints = build_renderer(paz.graphics.SPHERE, SE3.scaling, 0)
plot_jacobian(render_with_constraints, jp.array([1.0, 1.0, 1.0]))

render_with_constraints = build_renderer(
    paz.graphics.SPHERE, SE3.translation, 1
)
plot_jacobian(render_with_constraints, jp.array([0.0, 0.0, 0.0]))
