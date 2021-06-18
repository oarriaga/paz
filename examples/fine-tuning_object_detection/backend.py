from paz.backend.image.opencv_image import _CHANNELS_TO_FLAG
from paz.backend.image import convert_color_space
from paz.backend.image import BGR2RGB
import cv2

BGRA2RGBA = cv2.COLOR_BGRA2RGBA


def load_image(filepath, num_channels=3):
    """Load image from a ''filepath''.

    # Arguments
        filepath: String indicating full path to the image.
        num_channels: Int.

    # Returns
        Numpy array.
    """
    image = cv2.imread(filepath, _CHANNELS_TO_FLAG[num_channels])
    if num_channels == 1:
        return image
    elif num_channels == 3:
        return convert_color_space(image, BGR2RGB)
    elif num_channels == 4:
        return convert_color_space(image, BGRA2RGBA)
    else:
        raise ValueError('Invalid number of channels: ', num_channels)
