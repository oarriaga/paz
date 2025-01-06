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
