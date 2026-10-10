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


def to_rotation_vector(quaternion):
    """Maps a quaternion [w, x, y, z] to a rotation vector [3]."""
    # q and -q are the same rotation, a non-negative w keeps the angle in
    # [0, pi]
    is_flipped = get_real(quaternion) < 0.0
    quaternion = jp.where(is_flipped, -quaternion, quaternion)
    imaginary = get_imaginary(quaternion)
    norm_squared = jp.dot(imaginary, imaginary)
    ratio = compute_angle_ratio(norm_squared, get_real(quaternion))
    return ratio * imaginary


def compute_angular_rate(quaternion, quaternion_rate):
    """Maps a quaternion rate to the body-frame angular rate."""
    # q' = q (0, w) / 2 gives w = 2 Im(q* q'). Passing q'' gives the angular
    # acceleration since q* q'' = (w / 2)^2 + w' / 2 and (w / 2)^2 is real
    relative_rate = multiply(conjugate(quaternion), quaternion_rate)
    return 2.0 * get_imaginary(relative_rate)


def multiply(quaternion_A, quaternion_B):
    """Hamilton product of two quaternions [w, x, y, z]."""
    real_A = get_real(quaternion_A)
    real_B = get_real(quaternion_B)
    imaginary_A = get_imaginary(quaternion_A)
    imaginary_B = get_imaginary(quaternion_B)
    real = real_A * real_B - jp.dot(imaginary_A, imaginary_B)
    scaled = real_A * imaginary_B + real_B * imaginary_A
    imaginary = scaled + jp.cross(imaginary_A, imaginary_B)
    return jp.concatenate([real[None], imaginary])


def conjugate(quaternion):
    return quaternion * jp.array([1.0, -1.0, -1.0, -1.0])


def compute_angle_ratio(norm_squared, real):
    """Ratio `2 atan2(norm, real) / norm` of rotation angle to vector norm.

    It is `0 / 0` at `norm = 0` with the limit `2 / real`. When the
    tangent `t = norm / real` is small a Taylor series in `t^2` replaces
    the ratio, and safe values keep the unused branch free of NaN under
    autodiff.
    """
    use_taylor = norm_squared < 1e-8 * real**2
    safe_norm_squared = jp.where(use_taylor, 1.0, norm_squared)
    norm = jp.sqrt(safe_norm_squared)
    exact = 2.0 * jp.arctan2(norm, real) / norm
    safe_real = jp.where(use_taylor, real, 1.0)
    t_squared = norm_squared / safe_real**2
    series = 1.0 - t_squared / 3.0 + t_squared**2 / 5.0
    return jp.where(use_taylor, (2.0 / safe_real) * series, exact)
