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


def resize(image, shape):
    return jax.image.resize(image, (*shape, 3), "bilinear")


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
