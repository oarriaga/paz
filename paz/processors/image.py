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
from ..backend.image import blend_alpha_channel
from ..backend.image import random_shape_crop
from ..backend.image import make_random_plain_image
from ..backend.image import concatenate_alpha_mask
from ..backend.image import draw_filled_polygon
from ..backend.image import gaussian_image_blur
from ..backend.image import normalized_device_coordinates_to_image
from ..backend.image import image_to_normalized_device_coordinates
from ..backend.image import replace_lower_than_threshold
from ..backend.image import flip_left_right
from ..backend.image import BILINEAR, CUBIC
from ..backend.image.tensorflow_image import imagenet_preprocess_input


B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN = 104, 117, 123
BGR_IMAGENET_MEAN = (B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN)
RGB_IMAGENET_MEAN = (R_IMAGENET_MEAN, G_IMAGENET_MEAN, B_IMAGENET_MEAN)


class CastImage(Processor):
    """Cast image to given dtype.

    # Arguments
        dtype: Str or np.dtype
    """
    def __init__(self, dtype):
        self.dtype = dtype
        super(CastImage, self).__init__()

    def call(self, image):
        return cast_image(image, self.dtype)


class SubtractMeanImage(Processor):
    """Subtract channel-wise mean to image.

    # Arguments
        mean: List of length 3, containing the channel-wise mean.
    """
    def __init__(self, mean):
        self.mean = mean
        super(SubtractMeanImage, self).__init__()

    def call(self, image):
        return image - self.mean


class AddMeanImage(Processor):
    """Adds channel-wise mean to image.

    # Arguments
        mean: List of length 3, containing the channel-wise mean.
    """
    def __init__(self, mean):
        self.mean = mean
        super(AddMeanImage, self).__init__()

    def call(self, image):
        return image + self.mean


class NormalizeImage(Processor):
    """Normalize image by diving all values by 255.0.
    """
    def __init__(self):
        super(NormalizeImage, self).__init__()

    def call(self, image):
        return image / 255.0


class DenormalizeImage(Processor):
    """Denormalize image by multiplying all values by 255.0.
    """
    def __init__(self):
        super(DenormalizeImage, self).__init__()

    def call(self, image):
        return image * 255.0


class LoadImage(Processor):
    """Loads image.

    # Arguments
        num_channels: Integer, valid integers are: 1, 3 and 4.
    """
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        super(LoadImage, self).__init__()

    def call(self, image):
        return load_image(image, self.num_channels)


class RandomSaturation(Processor):
    """Applies random saturation to an image in RGB space.

    # Arguments
        lower: Float, lower bound for saturation factor.
        upper: Float, upper bound for saturation factor.
    """
    def __init__(self, lower=0.3, upper=1.5):
        self.lower = lower
        self.upper = upper
        super(RandomSaturation, self).__init__()

    def call(self, image):
        return random_saturation(image, self.lower, self.upper)


class RandomBrightness(Processor):
    """Adjust random brightness to an image in RGB space.

    # Arguments
        max_delta: Float.
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
        delta: Int, indicating the range (-delta, delta ) of possible
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
        size: List of two ints.
    """
    def __init__(self, shape, method=BILINEAR):
        self.shape = shape
        self.method = method
        super(ResizeImage, self).__init__()

    def call(self, image):
        return resize_image(image, self.shape, self.method)


class ResizeImages(Processor):
    """Resize list of images.

    # Arguments
        size: List of two ints.
    """
    def __init__(self, shape):
        self.shape = shape
        super(ResizeImages, self).__init__()

    def call(self, images):
        return [resize_image(image, self.shape) for image in images]


class RandomImageBlur(Processor):
    """Randomizes image quality

    # Arguments
        probability: Float between [0, 1]. Assigns probability of how
            often a random image blur is applied.
    """
    def __init__(self, probability=0.5):
        super(RandomImageBlur, self).__init__()
        self.probability = probability

    def call(self, image):
        if self.probability >= np.random.rand():
            image = random_image_blur(image)
        return image


class RandomGaussianBlur(Processor):
    """Randomizes image quality

    # Arguments
        probability: Float between [0, 1]. Assigns probability of how
            often a random image blur is applied.
    """
    def __init__(self, kernel_size=(5, 5), probability=0.5):
        super(RandomGaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.probability = probability

    def call(self, image):
        if self.probability >= np.random.rand():
            image = gaussian_image_blur(image, self.kernel_size)
        return image


class RandomFlipImageLeftRight(Processor):
    """Randomly flip the image left or right
    """
    def __init__(self):
        super(RandomFlipImageLeftRight, self).__init__()

    def call(self, image):
        return random_flip_left_right(image)


class ConvertColorSpace(Processor):
    """Converts image to a different color space.

    # Arguments
        flag: Flag found in ``processors``indicating transform e.g.
            ``pr.BGR2RGB``
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
    """Wrapper for Keras ImageDataGenerator

    # Arguments
        generator: An instantiated Keras ImageDataGenerator
    """
    def __init__(self, generator):
        super(ImageDataProcessor, self).__init__()
        self.generator = generator

    def call(self, image):
        random_parameters = self.generator.get_random_transform(image.shape)
        image = self.generator.apply_transform(image, random_parameters)
        image = self.generator.standardize(image)
        return image


class AlphaBlending(Processor):
    """Blends image to background using the image's alpha channel.
    """
    def __init__(self):
        super(AlphaBlending, self).__init__()

    def call(self, image, background):
        return blend_alpha_channel(image, background)


class RandomShapeCrop(Processor):
    """Randomly crops a part of an image of always the same given ``shape``.

    # Arguments
        shape: List of two ints [height, width].
            Dimensions of image to be cropped.
    """
    def __init__(self, shape):
        super(RandomShapeCrop, self).__init__()
        self.shape = shape

    def call(self, image):
        return random_shape_crop(image, self.shape)


class MakeRandomPlainImage(Processor):
    """Makes random plain image by randomly sampling an RGB color.

    # Arguments
        shape: List of two ints [height, width].
            Dimensions of plain image to be generated.
    """
    def __init__(self, shape):
        super(MakeRandomPlainImage, self).__init__()
        self.shape = shape

    def call(self):
        return make_random_plain_image(self.shape)


class ConcatenateAlphaMask(Processor):
    """Concatenates alpha mask to original image.
    """
    def __init__(self, **kwargs):
        super(ConcatenateAlphaMask, self).__init__(**kwargs)

    def call(self, image, alpha_mask):
        return concatenate_alpha_mask(image, alpha_mask)


class BlendRandomCroppedBackground(Processor):
    """Blends image with a randomly cropped background.

    # Arguments
        background_paths: List of strings. Each element of the list is a
            full-path to an image used for cropping a background.
    """
    def __init__(self, background_paths):
        super(BlendRandomCroppedBackground, self).__init__()
        if not isinstance(background_paths, list):
            raise ValueError('``background_paths`` must be list')
        if len(background_paths) == 0:
            raise ValueError('No paths given in ``background_paths``')
        self.background_paths = background_paths

    def call(self, image):
        random_arg = np.random.randint(0, len(self.background_paths))
        background_path = self.background_paths[random_arg]
        background = load_image(background_path)
        background = random_shape_crop(background, image.shape[:2])
        if background is None:
            H, W, num_channels = image.shape
            # background contains always a channel less
            num_channels = num_channels - 1
            background = make_random_plain_image((H, W, num_channels))
        return blend_alpha_channel(image, background)


class AddOcclusion(Processor):
    """Adds a random occlusion to image by generating random vertices and
        drawing a polygon.

    # Arguments
        max_radius_scale: Float between [0, 1].
            Value multiplied with largest image dimension to obtain the maximum
                radius possible of a vertex in the occlusion polygon.
        probability: Float between [0, 1]. Assigns probability of how
            often an occlusion to an image is generated.
    """
    def __init__(self, max_radius_scale=0.5, probability=0.5):
        super(AddOcclusion, self).__init__()
        self.max_radius_scale = max_radius_scale
        self.probability = probability

    def _random_vertices(self, center, max_radius, min_vertices, max_vertices):
        num_vertices = np.random.randint(min_vertices, max_vertices)
        angle_delta = 2 * np.pi / num_vertices
        initial_angle = np.random.uniform(0, 2 * np.pi)
        angles = initial_angle + np.arange(0, num_vertices) * angle_delta
        x_component = np.cos(angles).reshape(-1, 1)
        y_component = np.sin(angles).reshape(-1, 1)
        vertices = np.concatenate([x_component, y_component], -1)
        random_lengths = np.random.uniform(0, max_radius, num_vertices)
        random_lengths = random_lengths.reshape(num_vertices, 1)
        vertices = vertices * random_lengths
        vertices = vertices + center
        return vertices.astype(np.int32)

    def add_occlusion(self, image, max_radius_scale):
        height, width = image.shape[:2]
        max_radius = np.max((height, width)) * max_radius_scale
        center = np.random.rand(2) * np.array([width, height])
        vertices = self._random_vertices(center, max_radius, 3, 7)
        color = np.random.randint(0, 256, 3).tolist()
        return draw_filled_polygon(image, vertices, color)

    def call(self, image):
        if self.probability >= np.random.rand():
            image = self.add_occlusion(image, self.max_radius_scale)
        return image


class RandomImageCrop(Processor):
    """Crops randomly a rectangle from an image.

    # Arguments
        crop_factor: Float between ``[0, 1]``.
        probability: Float between ``[0, 1]``.
    """
    def __init__(self, crop_factor=0.3, probability=0.5):
        self.crop_factor = crop_factor
        self.probability = probability
        super(RandomImageCrop, self).__init__()

    def call(self, image):
        if self.probability < np.random.rand():
            return image
        H, W = image.shape[:2]
        W_crop = np.random.uniform(self.crop_factor * W, W)
        H_crop = np.random.uniform(self.crop_factor * H, H)
        x_min = np.random.uniform(W - W_crop)
        y_min = np.random.uniform(H - H_crop)
        x_max = x_min + W_crop
        y_max = y_min + H_crop
        cropped_image = image[int(x_min):int(x_max), int(y_min):int(y_max), :]
        return cropped_image


class ImageToNormalizedDeviceCoordinates(Processor):
    """Map image value from [0, 255] -> [-1, 1].
    """
    def __init__(self):
        super(ImageToNormalizedDeviceCoordinates, self).__init__()

    def call(self, image):
        return image_to_normalized_device_coordinates(image)


class NormalizedDeviceCoordinatesToImage(Processor):
    """Map normalized value from [-1, 1] -> [0, 255].
    """
    def __init__(self):
        super(NormalizedDeviceCoordinatesToImage, self).__init__()

    def call(self, image):
        return normalized_device_coordinates_to_image(image)


class ReplaceLowerThanThreshold(Processor):
    def __init__(self, threshold=1e-8, replacement=0.0):
        super(ReplaceLowerThanThreshold, self).__init__()
        self.threshold = threshold
        self.replacement = replacement

    def call(self, values):
        return replace_lower_than_threshold(
            values, self.threshold, self.replacement)


class GetNonZeroValues(Processor):
    def __init__(self):
        super(GetNonZeroValues, self).__init__()

    def call(self, array):
        channel_wise_sum = np.sum(array, axis=2)
        non_zero_arguments = np.nonzero(channel_wise_sum)
        return array[non_zero_arguments]


class GetNonZeroArguments(Processor):
    def __init__(self):
        super(GetNonZeroArguments, self).__init__()

    def call(self, array):
        channel_wise_sum = np.sum(array, axis=2)
        non_zero_rows, non_zero_columns = np.nonzero(channel_wise_sum)
        return non_zero_rows, non_zero_columns


class ImagenetPreprocessInput(Processor):
    def __init__(self):
        super(ImagenetPreprocessInput, self).__init__()

    def call(self, image):
        return imagenet_preprocess_input(image)


class FlipLeftRightImage(Processor):
    """Flips an image left and right.

    # Arguments
        image: Numpy array.
    """
    def __init__(self):
        super(FlipLeftRightImage, self).__init__()

    def call(self, image):
        return flip_left_right(image)
