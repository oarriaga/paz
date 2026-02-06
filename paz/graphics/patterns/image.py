import jax.numpy as jp


def compute_image_colors(u, v, image):
    H, W = image.shape[:2]
    v = 1 - v
    x = u * (W - 1)
    y = v * (H - 1)

    x = jp.squeeze(x, axis=1)
    y = jp.squeeze(y, axis=1)

    x = x.round().astype(jp.int32)
    y = y.round().astype(jp.int32)
    return image[y, x]


def compute_image_colors_bilinear(u, v, image):
    H, W = image.shape[:2]
    v = 1 - v
    x = u * (W - 1)
    y = v * (H - 1)

    x = jp.squeeze(x, axis=-1)
    y = jp.squeeze(y, axis=-1)

    x0 = jp.floor(x).astype(jp.int32)
    y0 = jp.floor(y).astype(jp.int32)
    x1 = jp.minimum(x0 + 1, W - 1)
    y1 = jp.minimum(y0 + 1, H - 1)

    x0 = jp.clip(x0, 0, W - 1)
    x1 = jp.clip(x1, 0, W - 1)
    y0 = jp.clip(y0, 0, H - 1)
    y1 = jp.clip(y1, 0, H - 1)

    wx = (x - x0).reshape(-1, 1)
    wy = (y - y0).reshape(-1, 1)

    top = (1.0 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1.0 - wx) * image[y1, x0] + wx * image[y1, x1]
    return (1.0 - wy) * top + wy * bottom
