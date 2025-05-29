import jax
import jax.numpy as jp
from jax import value_and_grad, jit, vmap


def normalize_quaternion(q):
    return q / jp.linalg.norm(q)


def quaternion_to_rotation_matrix(q):
    q = normalize_quaternion(q)
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    return jp.array(
        [
            [
                1 - 2 * (qj**2 + qk**2),
                2 * (qi * qj - qk * qr),
                2 * (qi * qk + qj * qr),
            ],
            [
                2 * (qi * qj + qk * qr),
                1 - 2 * (qi**2 + qk**2),
                2 * (qj * qk - qi * qr),
            ],
            [
                2 * (qi * qk - qj * qr),
                2 * (qj * qk + qi * qr),
                1 - 2 * (qi**2 + qj**2),
            ],
        ]
    )


def get_covariance_matrix(scale, rot_quat):
    # scale is a 3D vector, rot_quat is a 4D quaternion
    S = jp.diag(scale)
    R = quaternion_to_rotation_matrix(rot_quat)
    return R @ S @ S.T @ R.T


def project_gaussian(mean_3d, cov_3d, view_matrix, proj_matrix):
    # Convert to homogeneous coordinates
    mean_homogeneous = jp.append(mean_3d, 1.0)

    # Transform to camera space
    p_camera = view_matrix @ mean_homogeneous

    # Extract depth for sorting later
    depth = p_camera[2]

    # Project to clip space
    p_clip = proj_matrix @ p_camera

    # Perspective divide to get normalized device coordinates (NDC)
    p_ndc = p_clip[:2] / p_clip[3]

    # Jacobian of the affine approximation of the projective transformation
    J = jp.array(
        [
            [
                proj_matrix[0, 0] / p_camera[2],
                proj_matrix[0, 1] / p_camera[2],
                -proj_matrix[0, 0] * p_camera[0] / (p_camera[2] ** 2),
            ],
            [
                proj_matrix[1, 0] / p_camera[2],
                proj_matrix[1, 1] / p_camera[2],
                -proj_matrix[1, 1] * p_camera[1] / (p_camera[2] ** 2),
            ],
        ]
    )

    # Transform covariance to 2D using the viewing transformation and Jacobian
    viewing_transform = view_matrix[:3, :3]
    cov_2d = J @ viewing_transform @ cov_3d @ viewing_transform.T @ J.T

    return p_ndc, cov_2d, depth


def splat_pixel(pixel_coords, means_2d, covs_2d, colors, opacities, depths):
    # Sort Gaussians by depth (front to back)
    sorted_indices = jp.argsort(depths)

    means_2d_sorted = means_2d[sorted_indices]
    covs_2d_sorted = covs_2d[sorted_indices]
    colors_sorted = colors[sorted_indices]
    opacities_sorted = opacities[sorted_indices]

    # Initial pixel color and alpha
    pixel_color = jp.zeros(3)
    T = 1.0  # Transmittance

    # Alpha blend sorted Gaussians
    def blend_gaussian(carry, params):
        pixel_color_acc, T_acc = carry
        mean, cov, color, opacity = params

        # Gaussian evaluation at the pixel
        delta = pixel_coords - mean
        exponent = -0.5 * delta.T @ jp.linalg.inv(cov) @ delta
        alpha_pre = jp.exp(exponent)

        # Multiply by the learned opacity
        alpha = opacity * alpha_pre

        # Update color and transmittance
        new_pixel_color = pixel_color_acc + T_acc * alpha * color
        new_T = T_acc * (1.0 - alpha)

        return (new_pixel_color, new_T), None

    (pixel_color, _), _ = jax.lax.scan(
        blend_gaussian,
        (pixel_color, T),
        (means_2d_sorted, covs_2d_sorted, colors_sorted, opacities_sorted),
    )

    return pixel_color


def render_image(gaussians, view_matrix, proj_matrix, width, height):
    # Unpack gaussians
    means_3d = gaussians["means"]
    scales = gaussians["scales"]
    quats = gaussians["quats"]
    opacities = gaussians["opacities"]
    colors = gaussians["colors"]

    # Get 3D covariance for all Gaussians
    get_cov_vmap = vmap(get_covariance_matrix)
    covs_3d = get_cov_vmap(scales, quats)

    # Project all Gaussians to 2D
    project_vmap = vmap(project_gaussian, in_axes=(0, 0, None, None))
    means_2d, covs_2d, depths = project_vmap(
        means_3d, covs_3d, view_matrix, proj_matrix
    )

    # Convert from NDC to pixel coordinates
    def ndc_to_pixel(p_ndc):
        return jp.array(
            [(p_ndc[0] + 1) * 0.5 * width, (p_ndc[1] + 1) * 0.5 * height]
        )

    pixel_means = vmap(ndc_to_pixel)(means_2d)

    # Create pixel grid
    x = jp.arange(width)
    y = jp.arange(height)
    X, Y = jp.meshgrid(x, y)
    pixel_grid = jp.stack([X, Y], axis=-1).reshape(-1, 2)

    # Render each pixel
    render_vmap = vmap(splat_pixel, in_axes=(0, None, None, None, None, None))
    rendered_pixels = render_vmap(
        pixel_grid, pixel_means, covs_2d, colors, opacities, depths
    )

    return rendered_pixels.reshape(height, width, 3)


def loss_fn(gaussians, view_matrix, proj_matrix, gt_image):
    width, height = gt_image.shape[1], gt_image.shape[0]
    rendered_image = render_image(
        gaussians, view_matrix, proj_matrix, width, height
    )
    return jp.mean((rendered_image - gt_image) ** 2)


@jit
def train_step(gaussians, view_matrix, proj_matrix, gt_image, learning_rate):
    loss, grads = value_and_grad(loss_fn, has_aux=False)(
        gaussians, view_matrix, proj_matrix, gt_image
    )

    # Update parameters using basic gradient descent
    updated_gaussians = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, gaussians, grads
    )

    # Make sure opacities and scales stay in a valid range
    updated_gaussians["opacities"] = jp.clip(
        updated_gaussians["opacities"], 0.0, 1.0
    )
    updated_gaussians["scales"] = jax.nn.softplus(updated_gaussians["scales"])

    return updated_gaussians, loss


# --- Example Usage ---
if __name__ == "__main__":
    # Use a random key for initialization
    key = jax.random.PRNGKey(0)
    num_gaussians = 50

    # Initialize random Gaussians
    gaussians = {
        "means": jax.random.uniform(
            key, (num_gaussians, 3), minval=-1, maxval=1
        ),
        "scales": jax.random.uniform(
            key, (num_gaussians, 3), minval=0.1, maxval=0.5
        ),
        "quats": jax.random.uniform(
            key, (num_gaussians, 4), minval=-1, maxval=1
        ),
        "opacities": jax.random.uniform(
            key, (num_gaussians,), minval=0.5, maxval=0.9
        ),
        "colors": jax.random.uniform(
            key, (num_gaussians, 3), minval=0, maxval=1
        ),
    }

    # Example camera and image settings
    img_size = 64
    width, height = img_size, img_size

    # Dummy ground truth image (e.g., a green square)
    gt_image = jp.zeros((height, width, 3))
    gt_image = gt_image.at[16:48, 16:48, 1].set(
        1.0
    )  # Green square in the middle

    # Simple camera parameters (looking along the Z-axis)
    view_matrix = jp.eye(4).at[2, 3].set(-2.5)

    focal_length = 2.0
    proj_matrix = jp.array(
        [
            [focal_length, 0, 0, 0],
            [0, focal_length, 0, 0],
            [0, 0, -1, -0.1],
            [0, 0, -1, 0],
        ]
    )

    # Training loop
    learning_rate = 0.01
    for i in range(20_000):
        gaussians, loss = train_step(
            gaussians, view_matrix, proj_matrix, gt_image, learning_rate
        )
        if i % 20 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")

    print("Training complete.")
