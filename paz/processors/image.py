import numpy as np

from ..abstract import Processor

from ..backend.image import cast_image
from ..backend.image import load_image
from ..backend.image import random_saturation
from ..backend.image import random_brightness
from ..backend.image import random_contrast
from ..backend.image import random_hue
from ..backend.image import resize_image
from ..backend.image import random_image_blur
from ..backend.image import random_flip_left_right
from ..backend.image import convert_color_space
from ..backend.image import show_image


B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN = 104, 117, 123
BGR_IMAGENET_MEAN = (B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN)
RGB_IMAGENET_MEAN = (R_IMAGENET_MEAN, G_IMAGENET_MEAN, B_IMAGENET_MEAN)


class CastImage(Processor):
    """Cast image to given dtype.
    """
    def __init__(self, dtype):
        self.dtype = dtype
        super(CastImage, self).__init__()

    def call(self, image):
        return cast_image(image, self.dtype)


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
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        super(LoadImage, self).__init__()

    def call(self, image):
        return load_image(image, self.num_channels)


class RandomSaturation(Processor):
    """Applies random saturation to an image in RGB space.
    # Arguments
        lower: float. Lower bound for saturation factor.
        upper: float. Upper bound for saturation factor.
    """
    def __init__(self, lower=0.3, upper=1.5):
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
    def __init__(self, delta=32):
        self.delta = delta
        super(RandomBrightness, self).__init__()

    def call(self, image):
        return random_brightness(image, self.delta)


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
    def __init__(self, delta=18):
        self.delta = delta
        super(RandomHue, self).__init__()

    def call(self, image):
        return random_hue(image, self.delta)


class ResizeImage(Processor):
    """Resize image.
    # Arguments
        size: list of two ints.
    """
    def __init__(self, size):
        self.size = size
        super(ResizeImage, self).__init__()

    def call(self, image):
        return resize_image(image, self.size)


class ResizeImages(Processor):
    """Resize list of images.
    # Arguments
        size: list of two ints.
    """
    def __init__(self, size):
        self.size = size
        super(ResizeImages, self).__init__()

    def call(self, images):
        return [resize_image(image, self.shape) for image in images]


class RandomImageBlur(Processor):
    """Randomizes image quality
    """
    def __init__(self, probability=0.5):
        super(RandomImageBlur, self).__init__()
        self.probability = probability

    def call(self, image):
        if self.probability >= np.random.rand():
            image = random_image_blur(image)
        return image


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


class ShowImage(Processor):
    """Shows image in a separate window.
    # Arguments
        window_name: String. Window name.
        wait: Boolean
    """
    def __init__(self, window_name='image', wait=True):
        super(ShowImage, self).__init__()
        self.window_name = window_name
        self.wait = wait

    def call(self, image):
        return show_image(image, self.window_name, self.wait)


class ImageDataProcessor(Processor):
    def __init__(self, generator):
        super(ImageDataProcessor, self).__init__()
        self.generator = generator

    def call(self, image):
        random_parameters = self.generator.get_random_transform(image.shape)
        image = self.generator.apply_transform(image, random_parameters)
        image = self.generator.standardize(image)
        return image
