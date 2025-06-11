import cv2
import numpy as np
import jax
from jax.scipy.ndimage import map_coordinates
import jax.numpy as jp

import paz


B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN = 104, 117, 123
BGR_IMAGENET_MEAN = (B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN)
RGB_IMAGENET_MEAN = (R_IMAGENET_MEAN, G_IMAGENET_MEAN, B_IMAGENET_MEAN)
rgb_IMAGENET_MEAN = jp.array(
    [
        R_IMAGENET_MEAN / 255,
        G_IMAGENET_MEAN / 255,
        B_IMAGENET_MEAN / 255,
    ]
)
B_IMAGENET_STDV, G_IMAGENET_STDV, R_IMAGENET_STDV = 57.3, 57.1, 58.4
RGB_IMAGENET_STDV = (R_IMAGENET_STDV, G_IMAGENET_STDV, B_IMAGENET_STDV)
rgb_IMAGENET_STDV = jp.array(
    [
        R_IMAGENET_STDV / 255,
        G_IMAGENET_STDV / 255,
        B_IMAGENET_STDV / 255,
    ]
)

GRAY = cv2.IMREAD_GRAYSCALE
COLOR = cv2.IMREAD_COLOR
DEPTH = cv2.IMREAD_ANYDEPTH


def flip_left_right(image):
    """Flips an image left and right.

    # Arguments
        image: Array of shape `(H, W, C)`.

    # Returns
        Flipped image array.
    """
    return image[:, ::-1]


def BGR_to_RGB(image_BGR):
    return image_BGR[..., ::-1]


def RGB_to_BGR(image_RGB):
    return image_RGB[..., ::-1]


def load(filepath, flag=COLOR):
    image = jp.array(cv2.imread(filepath, flag))
    if flag == COLOR:
        image = BGR_to_RGB(image)
    elif flag == GRAY:
        image = jp.expand_dims(image, axis=-1)
    elif flag == DEPTH:
        image = jp.expand_dims(image, axis=-1)
    else:
        raise ValueError("Invalid flag")
    return image


def write(filepath, image):
    image = RGB_to_BGR(image)
    image = np.ascontiguousarray(paz.to_numpy(image))
    return cv2.imwrite(filepath, image)


def resize(image, size, method="linear", antialias=False):
    return jax.image.resize(image, (*size, image.shape[-1]), method, antialias)


def resize_opencv(image: jax.Array, size: tuple[int, int]) -> jax.Array:
    data = jax.ShapeDtypeStruct((size[0], size[1], image.shape[2]), image.dtype)

    def resize(image, shape):
        image = paz.to_numpy(image)
        image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_LINEAR)
        return image

    return jax.pure_callback(resize, data, image, size)


def scale(image, scale_factor, method="linear", antialias=False):
    H, W = get_size(image)
    H_scaled = int(H * scale_factor)
    W_scaled = int(W * scale_factor)
    return resize(image, (H_scaled, W_scaled), method, antialias)


def resize_with_aspect_ratio(
    image, largest_side, method="linear", antialias=False
):
    H, W = get_size(image)
    min_scale = min(largest_side / H, largest_side / W)
    return scale(image, min_scale, method, antialias)


def scale_with_aspect_ratio(image, scale, method="linear", antialias=False):
    H, W = get_size(image)
    H_scaled = int(H * scale[0])
    W_scaled = int(W * scale[1])
    resize(image, (H_scaled, W_scaled), method, antialias)


def show(image, name="image", wait=True):
    """Shows RGB image in an external window.

    # Arguments
        image: Numpy array
        name: String indicating the window name.
        wait: Boolean. If ''True'' window stays open until user presses a key.
            If ''False'' windows closes immediately.
    """
    image = paz.to_numpy(image)
    if image.dtype != np.uint8:
        raise ValueError(f"``image`` is type {image.dtype}. Must be ``uint8``")
    image = RGB_to_BGR(image)  # openCV default color space is BGR
    cv2.imshow(name, image)
    if wait:
        while True:
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


def normalize(image):
    return image / 255.0


def denormalize(image):
    return paz.cast(image * 255.0, jp.uint8)


def rgb_to_gray(image):
    rgb_weights = jp.array([0.2989, 0.5870, 0.1140])
    grayscale = jp.tensordot(image, rgb_weights, axes=(-1, -1))
    grayscale = jp.expand_dims(grayscale, axis=-1)
    return grayscale


def RGB_to_GRAY(image):
    image = normalize(image)
    image = rgb_to_gray(image)
    image = denormalize(image)
    return image


def preprocess(image, shape):
    return normalize(resize(image, shape))


def get_size(image):
    H, W = image.shape[:2]
    return H, W


def get_input_size(model):
    return model.input_shape[1:3]


def get_num_channels(image):
    return image.shape[-1]


def crop(image, box):
    x_min, y_min, x_max, y_max = box
    return image[y_min:y_max, x_min:x_max, :]


def crop_center(image, H_crop, W_crop):
    H_now, W_now = get_size(image)
    center_x = W_now // 2
    center_y = H_now // 2
    x_min = center_x - (W_crop // 2)
    y_min = center_y - (H_crop // 2)
    x_max = x_min + W_crop
    y_max = y_min + H_crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def standardize(image, mean, stdv):
    return (image - mean) / stdv


def normalize_min_max(x, axis=-1):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def random_brightness(key, image, delta=32):
    """Applies random brightness to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        delta: Int.
    """
    image = paz.cast(image, jp.float32)
    random_brightness = jax.random.uniform(key, (), jp.float32, -delta, delta)
    image = image + random_brightness
    image = jp.clip(image, 0, 255)
    image = paz.cast(image, jp.uint8)
    return image


def random_contrast(key, image, lower=0.5, upper=1.5):
    """Applies random contrast to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        lower: Float.
        upper: Float.
    """
    alpha = jax.random.uniform(key, (1,), jp.float32, lower, upper)
    image = paz.cast(image, jp.float32)
    image = image * alpha
    image = jp.clip(image, 0, 255)
    image = paz.cast(image, jp.uint8)
    return image


def split_channels(image, num_channels=3):
    channels = jp.split(image, num_channels, axis=-1)
    return tuple(jp.squeeze(channel, axis=-1) for channel in channels)


def merge_channels(channel_0, channel_1, channel_2):
    channel_0 = jp.expand_dims(channel_0, axis=-1)
    channel_1 = jp.expand_dims(channel_1, axis=-1)
    channel_2 = jp.expand_dims(channel_2, axis=-1)
    return jp.concatenate([channel_0, channel_1, channel_2], axis=-1)


def rgb_to_hsv(image):
    """Convert image from RGB to HSV."""
    r, g, b = split_channels(image)
    channels_max = jp.max(image, axis=-1)  # value = channels_max
    channels_min = jp.min(image, axis=-1)
    delta = channels_max - channels_min
    safe_value = jp.where(channels_max > 0, channels_max, 1.0)
    safe_delta = jp.where(delta > 0, delta, 1.0)
    saturation = jp.where(channels_max > 0, delta / safe_value, 0.0)
    norm = 1.0 / (6.0 * safe_delta)
    hue = jp.where(
        channels_max == g,
        norm * (b - r) + 2.0 / 6.0,
        norm * (r - g) + 4.0 / 6.0,
    )
    hue = jp.where(channels_max == r, norm * (g - b), hue)
    hue = jp.where(delta > 0, hue, 0.0) + (hue < 0.0)
    return merge_channels(hue, saturation, channels_max)


def hsv_to_rgb(image):
    """Converts hue, saturation, value planes to r, g, b color planes."""
    hue, saturation, value = split_channels(image)
    dh = (hue % 1.0) * 6.0  # Wrap when hue >= 360°.
    dr = jp.clip(jp.abs(dh - 3.0) - 1.0, 0.0, 1.0)
    dg = jp.clip(2.0 - jp.abs(dh - 2.0), 0.0, 1.0)
    db = jp.clip(2.0 - jp.abs(dh - 4.0), 0.0, 1.0)
    one_minus_saturation = 1.0 - saturation
    r = value * (one_minus_saturation + saturation * dr)
    g = value * (one_minus_saturation + saturation * dg)
    b = value * (one_minus_saturation + saturation * db)
    return merge_channels(r, g, b)


def random_saturation(key, image, lower=0.3, upper=1.5):
    """Applies random saturation to an RGB image."""
    image = normalize(image)
    image = rgb_to_hsv(image)
    h, s, v = split_channels(image)
    random_scale = jax.random.uniform(key, (), jp.float32, lower, upper)
    s = s * random_scale
    s = jp.clip(s, 0.0, 1.0)
    image = merge_channels(h, s, v)
    image = hsv_to_rgb(image)
    image = denormalize(image)
    return paz.cast(image, jp.uint8)


def random_hue(key, image, max_delta=0.1):
    """Applies random hue adjustment to an RGB image.

    Returns:
        Image array with adjusted hue, dtype uint8.
    """
    # max_delta must be in the interval [0, 0.5]
    image = normalize(image)
    image = rgb_to_hsv(image)
    h, s, v = split_channels(image)
    delta = jax.random.uniform(key, (), minval=-max_delta, maxval=max_delta)
    h = (h + delta) % 1.0
    image = merge_channels(h, s, v)
    image = hsv_to_rgb(image)
    image = denormalize(image)
    image = jp.clip(image, 0, 255)
    return image.astype(jp.uint8)


def random_color_transform(key, image):
    key_1, key_2, key_3, key_4 = jax.random.split(key, 4)
    image = random_saturation(key_1, image)
    image = random_brightness(key_2, image)
    image = random_contrast(key_3, image)
    image = random_hue(key_4, image)
    return image


def affine_transform(image, matrix, order=1, mode="nearest", cval=0.0):

    def build_image_indices(image):
        dimension_indices = [jp.arange(size) for size in image.shape]
        meshgrid = jp.meshgrid(*dimension_indices, indexing="ij")
        meshgrid = [jp.expand_dims(x, axis=-1) for x in meshgrid]
        indices = jp.concatenate(meshgrid, axis=-1)
        return indices

    offset = matrix[: image.ndim, image.ndim]
    matrix = matrix[: image.ndim, : image.ndim]
    coordinates = build_image_indices(image) @ matrix.T
    coordinates = jp.moveaxis(coordinates, source=-1, destination=0)
    offset = jp.full((3,), fill_value=offset)
    coordinates += jp.reshape(offset, (*offset.shape, 1, 1, 1))
    return map_coordinates(image, coordinates, order, mode, cval)


def rotate(image, angle, order=1, mode="nearest", cval=0.0):
    """Rotates an image around its center using interpolation."""
    rotation = paz.SO3.rotation_z(angle)
    image_center = (jp.asarray(image.shape) - 1.0) / 2.0
    translation = image_center - rotation @ image_center
    matrix = paz.SE3.to_affine_matrix(rotation, translation)
    return affine_transform(image, matrix, order=order, mode=mode, cval=cval)


def random_rotation(
    key, image, min_angle, max_angle, order=1, mode="nearest", cval=0.0
):
    angle = jax.random.uniform(key, (), minval=min_angle, maxval=max_angle)
    return rotate(image, angle, order, mode, cval)


def random_flip_left_right(key, image):
    do_flip = jax.random.bernoulli(key)
    return jax.lax.cond(do_flip, flip_left_right, lambda x: x, image)


def pyramid(image, scales=[1.0, 0.8, 0.64, 0.512, 0.4096, 0.3277, 0.2621]):
    H, W = paz.image.get_size(image)

    pyramid = []
    for scale in sorted(scales, reverse=True):
        pyramid_H = int(round(H * scale))
        pyramid_W = int(round(W * scale))
        scaled_image = paz.image.resize(image, (pyramid_H, pyramid_W))
        pyramid.append(scaled_image)
    return pyramid


def pad(image, top, bottom, left, right, mode="reflect", pad_value=0):
    pad_widths = ((top, bottom), (left, right), (0, 0))
    if mode == "constant":
        image = jp.pad(image, pad_widths, mode=mode, constant_values=pad_value)
    else:
        image = jp.pad(image, pad_widths, mode=mode)
    return image


def get_patch_shape(H, W, patch_size, strides, padding="valid"):
    H_patch, W_patch = patch_size
    y_stride, x_stride = strides
    if padding == "same":
        num_patch_rows = (H + y_stride - 1) // y_stride
        num_patch_cols = (W + x_stride - 1) // x_stride
    elif padding == "valid":
        num_patch_rows = (H - H_patch) // y_stride + 1
        num_patch_cols = (W - W_patch) // x_stride + 1
    else:
        raise ValueError(f"Unknown padding type: {padding}")
    return num_patch_rows, num_patch_cols


def pad_same(image, patch_size, strides):

    def get_patch_span(num_patches, stride_size, patch_size):
        num_stride_steps = num_patches - 1
        stride_distance = num_stride_steps * stride_size
        total_covered_distance = stride_distance + patch_size
        return total_covered_distance

    def compute_cover(H, W, patch_size, strides):
        patch_shape = get_patch_shape(H, W, patch_size, strides, "same")
        H_covered = get_patch_span(patch_shape[0], strides[0], patch_size[0])
        W_covered = get_patch_span(patch_shape[1], strides[1], patch_size[1])
        return H_covered, W_covered

    def compute_needed_pad(covered_size, original_size):
        total_residue = covered_size - original_size
        minor_half_residue = total_residue // 2
        major_half_residue = total_residue - minor_half_residue
        return minor_half_residue, major_half_residue

    H, W = paz.image.get_size(image)
    H_covered, W_covered = compute_cover(H, W, patch_size, strides)
    y_minor_pad, y_major_pad = compute_needed_pad(H_covered, H)
    x_minor_pad, x_major_pad = compute_needed_pad(W_covered, W)
    pad_sizes = (y_minor_pad, y_major_pad, x_minor_pad, x_major_pad)
    return paz.image.pad(image, *pad_sizes)


def patch(image, patch_size, strides, padding="valid"):

    def build_min_args(H, W, patch_size, strides):
        H_patch, W_patch = patch_size
        y_stride, x_stride = strides
        y_min_args = jp.arange(H) * y_stride
        x_min_args = jp.arange(W) * x_stride
        return y_min_args, x_min_args

    def vectorized_patch(image, y_min_args, x_min_args, patch_size):
        H_patch, W_patch = patch_size

        def patch_one(y_min_args, x_min_args):
            start_args = (y_min_args, x_min_args, 0)
            slice_args = (H_patch, W_patch, get_num_channels(image))
            return jax.lax.dynamic_slice(image, start_args, slice_args)

        def patch_rows(y_min_args):
            return jax.vmap(patch_one, (None, 0))(y_min_args, x_min_args)

        return jax.vmap(patch_rows)(y_min_args)

    H, W = paz.image.get_size(image)
    H_out, W_out = get_patch_shape(H, W, patch_size, strides, padding)
    y_min_args, x_min_args = build_min_args(H_out, W_out, patch_size, strides)
    if padding == "same":
        image = paz.image.pad_same(image, patch_size, strides)
    return vectorized_patch(image, y_min_args, x_min_args, patch_size)


def augment_color(key, image):
    keys = jax.random.split(key, 4)
    image = paz.image.random_saturation(keys[0], image)
    image = paz.image.random_brightness(keys[1], image)
    image = paz.image.random_contrast(keys[2], image)
    image = paz.image.random_hue(keys[3], image)
    return image


def subtract_mean(image, mean):
    return image - mean


def divide_by_std(image, stdv):
    return image / stdv


def comput_aspect_ratio(image):
    H, W = paz.image.get_size(image)
    return W / H


def resize_pad_top_left(image, largest_side, method="linear", antialias=False):
    # TODO refactor to be able to resize any shape (H, W)
    """Resizes and crops image by returning the scales to original"""
    image = resize_with_aspect_ratio(image, largest_side, method, antialias)
    H, W = get_size(image)
    return pad(image, 0, largest_side - H, 0, largest_side - W, "constant", 0)
