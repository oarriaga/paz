import jax.numpy as jp


def add_one(point):
    return jp.concatenate([point, jp.ones(1)]).reshape(-1, 1)


def add_ones(points):
    ones = jp.ones((len(points), 1))
    points = jp.concatenate([points, ones], axis=-1)
    return points


def transform_points(affine_transform, points):
    points = add_ones(points)
    points = jp.matmul(affine_transform, points.T).T
    return points[:, :3]


def dehomogenize_coordinates(homogenous_point):
    homogenous_point = jp.squeeze(homogenous_point, axis=1)
    u, v, w = homogenous_point
    return jp.array([u / w, v / w])


def to_column_vector(vector):
    return vector.reshape(-1, 1)


def divide(numerator, denominator):
    add_epsilon_if_denom_zero_else_add_zero = 1e-6 * (denominator == 0.0)
    return numerator / (denominator + add_epsilon_if_denom_zero_else_add_zero)


def near_zero(scalar, epsilon=1e-6):
    """Determines whether a scalar is small enough to be treated as zero

    # Arguments:
        A scalar input to check

    # Returns:
        True if z is close to zero, false otherwise
    """
    return abs(scalar) < epsilon


def safe_norm(x, axis=None):
    """Calculates a linalg.norm(x) that's safe for gradients at x=0.
    Avoids a poorly defined gradient for jp.linalg.norm(0) see [1]

    # Arguments:
        x: A jnp.array
        axis: The axis along which to compute the norm

    # Returns:
        Norm of the array x.

    # References
       [1] https://github.com/google/jax/issues/3058 for details
    """

    is_zero = jp.allclose(x, 0.0)
    # temporarily swap x with ones if is_zero, then swap back
    x = jp.where(is_zero, jp.ones_like(x), x)
    norm = jp.linalg.norm(x, axis=axis)
    norm = jp.where(is_zero, 0.0, norm)
    return norm
