import cv2
import numpy as np
import jax
import jax.numpy as jp

import paz


BILINEAR = cv2.INTER_LINEAR


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


def load(filepath):
    return jp.array(BGR_to_RGB(cv2.imread(filepath)))


def resize(image, shape, method="bilinear"):
    return jax.image.resize(image, (*shape, image.shape[-1]), method)


def scale(image, scale, method="bilinear"):
    H, W, num_channels = image.shape
    H_scaled = int(H * scale[0])
    W_scaled = int(W * scale[1])
    return jax.image.resize(image, (H_scaled, W_scaled, num_channels), method)


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
        raise ValueError("``image`` must be of type ``uint8``")
    image = RGB_to_BGR(image)  # openCV default color space is BGR
    cv2.imshow(name, image)
    if wait:
        while True:
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


def normalize(image):
    return image / 255.0


def rgb_to_gray(image):
    weights = jp.array([0.2989, 0.5870, 0.1140], dtype=image.dtype)
    gray_image = jp.tensordot(image, weights, axes=(-1, -1))
    return jp.expand_dims(gray_image, axis=-1)
