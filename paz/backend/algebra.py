import jax.numpy as jp


def add_one(point):
    # TODO should not convert to column vector
    return jp.concatenate([point, jp.ones(1)]).reshape(-1, 1)


def add_ones(points):
    ones = jp.ones((len(points), 1))
    points = jp.concatenate([points, ones], axis=-1)
    return points


def transform_points(affine_transform, points):
    dimension = points.shape[-1]
    points = add_ones(points)
    points = jp.matmul(affine_transform, points.T).T
    return points[:, :dimension]


def transform(affine_trasnform, point):
    dimension = len(point)
    point = add_one(point)
    point = jp.squeeze(jp.matmul(affine_trasnform, point).T)
    point = point[:dimension]
    return point


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


def normalize_and_norm(x, axis=None):
    """Normalizes an array.

    # Arguments:
        x: A jnp.array
        axis: The axis along which to compute the norm

    # Returns:
        A tuple of (normalized array x, the norm).
    """
    norm = safe_norm(x, axis=axis)
    x_normalized = x / (norm + 1e-6 * (norm == 0.0))
    return x_normalized, norm


def compute_norms(vectors, axis=-1, keepdims=True):
    return jp.linalg.norm(vectors, axis=axis, keepdims=keepdims)


def normalize(vectors, axis=-1, keepdims=True):
    """Normalizes vectors across last dimension

    # Arguments
        vectors: Array (num_vectors, 3)

    # Returns
        normalized vectors: Array (num_vectors, 3)
    """
    norms = compute_norms(vectors, axis=axis, keepdims=keepdims)
    vectors = vectors / (norms + 1e-5)
    return vectors


def normalize_old(x, axis=None):
    """Normalizes an array.

    # Arguments:
        x: A jnp.array
        axis: The axis along which to compute the norm

    # Returns:
        A tuple of (normalized array x, the norm).
    """
    norm = safe_norm(x, axis=axis)
    x_normalized = x / (norm + 1e-6 * (norm == 0.0))
    return x_normalized


def dot(vectors_A, vectors_B):
    """Computes dot product between vectors_A and vectors_B

    # Arguments
        vectors_A: Array (num_vectors, 3)
        vectors_B: Array (num_vectors, 3)

    # Returns
        Array (num_rays)
    """
    return jp.sum(vectors_A * vectors_B, axis=-1)


def solve_quadratic(a, b, c):
    """Solves quadratic equation

    # Arguments
        a: Array
        b: Array
        c: Array

    # Returns
        solution_A: Array
        solution_B: Array
        valid_mask: Boolean array
    """
    discriminator = (b**2) - (4.0 * a * c)
    valid_mask = discriminator > 0  # >= is bad for automatic differentiation
    # TODO check if 0.0 should be epislon. Scipy optimize complained.
    discriminator = jp.where(valid_mask, discriminator, 1e-4)
    sqrt_discriminator = jp.sqrt(discriminator)
    solution_A = (-b - sqrt_discriminator) / (2.0 * a)
    solution_B = (-b + sqrt_discriminator) / (2.0 * a)
    return solution_A, solution_B, valid_mask


def add_zeros(vectors):
    """Adds zeros to vectors in R^3 across last dimension

    # Arguments
        vectors: Array (num_vectors, 3)

    # Returns
        vectors: Array (num_vectors, 4)
    """
    zeros = jp.zeros((len(vectors), 1))
    vectors = jp.concatenate([vectors, zeros], axis=-1)
    return vectors


def transform_vectors(affine_matrix, vectors):
    """Transform R^3 vectors with affine matrix

    # Arguments
        affine_matrix: (4, 4)
        rays: (num_rays, 3)

    # Returns
        Transformed vectors (num_rays, 3)
    """
    vectors = add_zeros(vectors)
    vectors = jp.matmul(affine_matrix, vectors.T).T
    vectors = vectors[:, :3]
    return vectors


def to_column(vector):
    return jp.reshape(vector, (-1, 1))
