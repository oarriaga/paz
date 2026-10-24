import jax.numpy as jp


def to_matrix(quaternion):
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    w, x, y, z = quaternion
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    xs = x * s
    ys = y * s
    zs = z * s
    wxs = w * xs
    wys = w * ys
    wzs = w * zs
    xxs = x * xs
    xys = x * ys
    xzs = x * zs
    yys = y * ys
    yzs = y * zs
    zzs = z * zs
    return jp.array(
        [
            [1.0 - (yys + zzs), xys - wzs, xzs + wys],
            [xys + wzs, 1.0 - (xxs + zzs), yzs - wxs],
            [xzs - wys, yzs + wxs, 1.0 - (xxs + yys)],
        ]
    )


def from_rotation_vector(rotation_vector):
    """Transforms rotation vector into quaternion.

    # Arguments
        rotation_vector: Array of shape ``[3]``.

    # Returns
        Array representing a quaternion having a shape ``[4]``.
    """
    theta = jp.linalg.norm(rotation_vector)
    rotation_axis = rotation_vector / theta
    half_theta = 0.5 * theta
    norm = jp.sin(half_theta)
    quaternion = jp.array(
        [
            norm * rotation_axis[0],
            norm * rotation_axis[1],
            norm * rotation_axis[2],
            jp.cos(half_theta),
        ]
    )
    return quaternion


def from_matrix(SO3_matrix):
    """Transforms rotation matrix into a ``[w, x, y, z]`` quaternion."""
    m00, m01, m02 = SO3_matrix[0]
    m10, m11, m12 = SO3_matrix[1]
    m20, m21, m22 = SO3_matrix[2]
    traces = jp.array(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ]
    )
    magnitudes = jp.sqrt(jp.maximum(traces, 0.0))
    candidates = jp.array(
        [
            [magnitudes[0] ** 2, m21 - m12, m02 - m20, m10 - m01],
            [m21 - m12, magnitudes[1] ** 2, m10 + m01, m02 + m20],
            [m02 - m20, m10 + m01, magnitudes[2] ** 2, m12 + m21],
            [m10 - m01, m20 + m02, m21 + m12, magnitudes[3] ** 2],
        ]
    )
    # The largest magnitude keeps the division well conditioned.
    candidates = candidates / (2.0 * jp.maximum(magnitudes, 0.1)[:, None])
    return candidates[jp.argmax(magnitudes)]


def slerp(quaternion_0, quaternion_1, ratio):
    """Interpolates two ``[w, x, y, z]`` quaternions along the short arc."""
    cosine = jp.dot(quaternion_0, quaternion_1)
    quaternion_1 = jp.where(cosine < 0.0, -quaternion_1, quaternion_1)
    angle = jp.arccos(jp.clip(jp.abs(cosine), -1.0, 1.0))
    sine = jp.maximum(jp.sin(angle), 1e-8)
    weight_0 = jp.sin((1.0 - ratio) * angle) / sine
    weight_1 = jp.sin(ratio * angle) / sine
    is_close = angle < 1e-6
    weight_0 = jp.where(is_close, 1.0 - ratio, weight_0)
    weight_1 = jp.where(is_close, ratio, weight_1)
    quaternion = weight_0 * quaternion_0 + weight_1 * quaternion_1
    return quaternion / jp.linalg.norm(quaternion)
