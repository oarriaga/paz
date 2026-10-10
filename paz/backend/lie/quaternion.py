import jax
import jax.numpy as jp

from paz.backend.lie import SO3

# Quaternions are arrays [w, x, y, z] with the real part w first.


def get_real(quaternion):
    return quaternion[0]


def get_imaginary(quaternion):
    return quaternion[1:4]


def wxyz_to_xyzw(quaternion):
    return jp.roll(jp.asarray(quaternion), -1)


def xyzw_to_wxyz(quaternion):
    return jp.roll(jp.asarray(quaternion), 1)


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
    """Maps a rotation vector [3] to a quaternion [w, x, y, z]."""
    # both parts depend only on the squared half angle so the value and
    # its first two derivatives stay finite at the zero rotation
    half_angle_squared = 0.25 * jp.dot(rotation_vector, rotation_vector)
    versine_ratio = SO3.compute_versine_ratio(half_angle_squared)
    real = 1.0 - half_angle_squared * versine_ratio
    sinc = SO3.compute_sinc(half_angle_squared)
    imaginary = 0.5 * sinc * rotation_vector
    return jp.concatenate([real[None], imaginary])


def compute_rates(rotation_vector, rate, acceleration):
    """Returns the quaternion and its first two time derivatives."""

    def compute_rate(rotation_vector, rate):
        args = (rotation_vector,), (rate,)
        return jax.jvp(from_rotation_vector, *args)[1]

    args = (rotation_vector,), (rate,)
    quaternion, quaternion_rate = jax.jvp(from_rotation_vector, *args)
    args = (rotation_vector, rate), (rate, acceleration)
    _, quaternion_acceleration = jax.jvp(compute_rate, *args)
    return quaternion, quaternion_rate, quaternion_acceleration
