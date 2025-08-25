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
