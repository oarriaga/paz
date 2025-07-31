import jax
import jax.numpy as jp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


def generate_data(num_inliers, num_outliers, key):
    """
    Generates a synthetic 3D point cloud with inliers on a plane and outliers.

    Args:
        num_inliers (int): The number of inlier points to generate close to the plane.
        num_outliers (int): The number of outlier points to generate far from the plane.
        key (jax.random.PRNGKey): The JAX random key for reproducibility.

    Returns:
        A tuple containing:
        - The combined point cloud (jax.numpy.ndarray).
        - The true normal vector of the plane (jax.numpy.ndarray).
        - The true offset of the plane (float).
        - The inlier points (jax.numpy.ndarray).
        - The outlier points (jax.numpy.ndarray).
    """
    # 1. Define the true plane's parameters
    # The true normal vector (must be a unit vector for a unique representation)
    true_normal = jp.array([0.1, 0.2, 0.95])
    true_normal /= jp.linalg.norm(true_normal)
    # The true offset distance from the origin
    true_offset = 0.5

    # 2. Generate inlier points
    key, subkey1, subkey2 = jax.random.split(key, 3)
    # Generate random points in the XY plane
    inlier_xy = jax.random.uniform(
        subkey1, shape=(num_inliers, 2), minval=-10, maxval=10
    )
    # Calculate their z-coordinate to make them lie exactly on the plane
    # Equation: n_x*x + n_y*y + n_z*z + d = 0  =>  z = (-n_x*x - n_y*y - d) / n_z
    inlier_z = (
        -true_normal[0] * inlier_xy[:, 0]
        - true_normal[1] * inlier_xy[:, 1]
        - true_offset
    ) / true_normal[2]

    inliers = jp.hstack([inlier_xy, inlier_z[:, None]])

    # Add some noise to the inliers to simulate real-world measurement errors
    # We use a Student's T distribution for noise to have some "heavier" small errors.
    noise_dist = tfd.StudentT(df=4, loc=0.0, scale=0.1)
    noise = noise_dist.sample(sample_shape=inliers.shape, seed=subkey2)
    inliers += noise

    # 3. Generate outlier points randomly in a larger volume
    key, subkey3 = jax.random.split(key, 2)
    outliers = jax.random.uniform(
        subkey3, shape=(num_outliers, 3), minval=-20, maxval=20
    )

    # 4. Combine inliers and outliers into a single point cloud
    points = jp.vstack([inliers, outliers])

    return points, true_normal, true_offset, inliers, outliers
