import numpy as np

from ..abstract import Processor

from ..backend.image import cast_image
from ..backend.image import load_image
from ..backend.image import random_saturation
from ..backend.image import random_brightness
from ..backend.image import random_contrast
from ..backend.image import random_hue
from ..backend.image import resize
from ..backend.image import random_image_quality
from ..backend.image import random_flip_left_right
from ..backend.image import convert_color_space

from ..backend.image import random_crop
from ..backend.image import random_plain_background
from ..backend.image import random_cropped_background
from ..backend.image import draw_random_polygon
from ..backend.image import show_image


B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN = 104, 117, 123
BGR_IMAGENET_MEAN = (B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN)
RGB_IMAGENET_MEAN = (R_IMAGENET_MEAN, G_IMAGENET_MEAN, B_IMAGENET_MEAN)


class CastImage(Processor):
    """Cast image to given dtype.
    """
    def __init__(self, dtype):
        self.dtype = self.dtype

    def call(self, image):
        cast_image(image, self.dtype)


class SubtractMeanImage(Processor):
    """Subtract channel-wise mean to image.
    # Arguments
        mean. List of length 3, containing the channel-wise mean.
    """
    def __init__(self, mean):
        self.mean = mean
        super(SubtractMeanImage, self).__init__()

    def call(self, image):
        return image - self.mean


class AddMeanImage(Processor):
    """Subtract channel-wise mean to image.
    # Arguments
        mean. List of length 3, containing the channel-wise mean.
    """
    def __init__(self, mean):
        self.mean = mean
        super(AddMeanImage, self).__init__()

    def call(self, image):
        return image + self.mean


class NormalizeImage(Processor):
    """Normalize image by diving its values by 255.0.
    """
    def __init__(self):
        super(NormalizeImage, self).__init__()

    def call(self, image):
        return image / 255.0


class DenormalizeImage(Processor):
    """Denormalize image by diving its values by 255.0.
    """
    def __init__(self):
        super(DenormalizeImage, self).__init__()

    def call(self, image):
        return image * 255.0


class LoadImage(Processor):
    """Decodes image filepath and loads image as tensor.
    # TODO: Raise value error whenever the image was not found.
    """
    def __init__(self, num_channels):
        self.num_channels = num_channels
        super(LoadImage, self).__init__()

    def call(self, filepath):
        return load_image(filepath, self.num_channels)


class RandomSaturation(Processor):
    """Applies random saturation to an image in RGB space.
    # Arguments
        lower: float. Lower bound for saturation factor.
        upper: float. Upper bound for saturation factor.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        super(RandomSaturation, self).__init__()

    def call(self, image):
        return random_saturation(image, self.lower, self.upper)


class RandomBrightness(Processor):
    """Adjust random brightness to an image in RGB space.
    # Arguments:
        max_delta: float.
    """
    def __init__(self, max_delta=32):
        self.max_delta = max_delta
        super(RandomBrightness, self).__init__()

    def call(self, image):
        return random_brightness(image, self.max_delta)


class RandomContrast(Processor):
    """Applies random contrast to an image in RGB
    # Arguments
        lower: Float, indicating the lower bound of the random number
            to be multiplied with the BGR/RGB image.
        upper: Float, indicating the upper bound of the random number
        to be multiplied with the BGR/RGB image.
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        super(RandomContrast, self).__init__()

    def call(self, image):
        return random_contrast(image, self.lower, self.upper)


class RandomHue(Processor):
    """Applies random hue to an image in RGB space.
    # Arguments
        delta: Integer, indicating the range (-delta, delta ) of possible
            hue values.
    """
    def __init__(self, max_delta=18):
        self.max_delta = max_delta
        super(RandomHue, self).__init__()

    def call(self, image):
        return random_hue(image, self.max_delta)


class ResizeImage(Processor):
    """Resize image.
    # Arguments
        size: list of two ints.
    """
    def __init__(self, size):
        self.size = size
        super(ResizeImage, self).__init__()

    def call(self, image):
        return resize(image, self.size)


class ResizeImages(Processor):
    """Resize list of images.
    # Arguments
        size: list of two ints.
    """
    def __init__(self, size):
        self.size = size
        super(ResizeImages, self).__init__()

    def call(self, images):
        return [resize(image, self.shape) for image in images]


class RandomImageQuality(Processor):
    """Randomizes image quality
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        super(RandomImageQuality, self).__init__()

    def call(self, image):
        return random_image_quality(image, self.lower, self.upper)


class RandomFlipImageLeftRight(Processor):
    def __init__(self):
        super(RandomFlipImageLeftRight, self).__init__()

    def call(self, image):
        return random_flip_left_right(image)


class ConvertColorSpace(Processor):
    """Converts image to a different color space.
    # Arguments
        flag: Flag found in ``ops``indicating transform e.g. BGR2RGB
    """
    def __init__(self, flag):
        self.flag = flag
        super(ConvertColorSpace, self).__init__()

    def call(self, image):
        return convert_color_space(image, self.flag)


class RandomPlainBackground(Processor):
    """Adds a monochromatic background to an image with an alpha-mask channel.
    """
    def __init__(self):
        super(RandomPlainBackground, self).__init__()

    def call(self, image):
        return random_plain_background(image)


class RandomImageCrop(Processor):
    """Crops and returns random patch of an image.
    """
    def __init__(self, size):
        self.size = size
        super(RandomImageCrop, self).__init__()

    def call(self, image):
        random_crop(image, self.size)


class RandomCroppedBackground(Processor):
    """Add a random cropped background from a randomly selected given set of
    images to a .png image with an alpha-mask channel.
    # Arguments:
        image_filepaths: List containing the full path to the images
            used to randomly select a crop.
    """
    def __init__(self, image_filepaths):
        self.image_filepaths = image_filepaths
        super(RandomCroppedBackground, self).__init__()

    def call(self, image):
        random_arg = np.random.randint(0, len(self.image_filepaths))
        background = load_image(self.image_filepaths[random_arg])
        return random_cropped_background(image, background)


class DrawRandomPolygon(Processor):
    """ Adds occlusion to image
    # Arguments
        max_radius_scale: Maximum radius in scale with respect to image i.e.
            each vertex radius from the polygon is sampled
            from [0, max_radius_scale]. This radius is later multiplied by
            the image dimensions.
    """
    def __init__(self, max_radius_scale=.5):
        super(DrawRandomPolygon, self).__init__()
        self.max_radius_scale = max_radius_scale

    def call(self, image):
        return draw_random_polygon(image)


class ShowImage(Processor):
    """Shows image in a separate window.
    # Arguments
        window_name: String. Window name.
        wait: Boolean
    """
    def __init__(self, window_name='image', wait=True):
        self.window_name = window_name,
        self.wait
        super(ShowImage, self).__init__()

    def call(self, image):
        return show_image(image, self.window_name, self.wait)
