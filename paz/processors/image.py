# References: https://github.com/amdegroot/ssd.pytorch

import numpy as np
from numpy import random
from ..core import ops

from ..core import Processor

B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN = 104, 117, 123
BGR_IMAGENET_MEAN = (B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN)
RGB_IMAGENET_MEAN = (R_IMAGENET_MEAN, G_IMAGENET_MEAN, B_IMAGENET_MEAN)


class CastImageToFloat(Processor):
    """Cast image to float32.
    """
    def __init__(self):
        super(CastImageToFloat, self).__init__()

    def call(self, image):
        return image.astype(np.float32)


class CastImageToInts(Processor):
    """Cast image to uint8
    """
    def __init__(self):
        super(CastImageToInts, self).__init__()

    def call(self, image):
        return np.round(image, decimals=0).astype(np.uint8)


class SubtractMeanImage(Processor):
    """Subtract channel-wise mean to image.
    # Arguments
        mean. List of length 3, containing the channel-wise mean.
    """
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
        super(SubtractMeanImage, self).__init__()

    def call(self, image):
        return image.astype(np.float32) - self.mean


class AddMeanImage(Processor):
    """Subtract channel-wise mean to image.
    # Arguments
        mean. List of length 3, containing the channel-wise mean.
    """
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
        super(AddMeanImage, self).__init__()

    def call(self, image):
        return image.astype(np.float32) + self.mean


class NormalizeImage(Processor):
    """Normalize image by diving its values by 255.0.
    """
    def __init__(self):
        super(NormalizeImage, self).__init__()

    def call(self, image):
        return image.astype(np.float32) / 255.0


class DenormalizeImage(Processor):
    """Denormalize image by diving its values by 255.0.
    """
    def __init__(self):
        super(DenormalizeImage, self).__init__()

    def call(self, image):
        return image * 255


class LoadImage(Processor):
    """Load image as numpy array.
    """
    def __init__(self, flags=ops.IMREAD_COLOR):
        self.flags = flags
        super(LoadImage, self).__init__()

    def call(self, filepath):
        image = ops.load_image(filepath, self.flags)
        if image is None:
            raise ValueError('Image not found in:', filepath)
        return image


class AddPlainBackground(Processor):
    """Add a monochromatic background to .png image with an alpha-mask channel.
    # Arguments:
        background_color: List or `None`.
            If `None` the background colors are filled randomly.
            If List, it should provide 3 int values representing BGR colors.
    """
    def __init__(self, background_color=None):
        self.background_color = background_color
        super(AddPlainBackground, self).__init__()

    def call(self, image):
        if image.shape[-1] != 4:
            raise ValueError('Provided image does not contain alpha mask.')
        H, W = image.shape[:2]
        image_shape = (H, W, 3)
        foreground, alpha = np.split(image, [3], -1)
        alpha = alpha / 255
        if self.background_color is None:
            background = np.ones((image_shape)) * random.randint(0, 256, 3)
            background = (1.0 - alpha) * background.astype(float)
        else:
            background_color = np.asarray(self.background_color)
            background = np.ones((image_shape)) * background_color
        foreground = alpha * foreground
        image = foreground + background
        return image


class AddCroppedBackground(Processor):
    """Add a random cropped background from a randomly selected given set of
    images to a .png image with an alpha-mask channel.
    # Arguments:
        image_filepaths: List containing the full path to the images
            used to randomly select a crop.
        box_size: List, containing the width and height of the
            background crop size.
    """
    def __init__(self, image_filepaths, box_size, color_model='BGR'):
        self.image_filepaths = image_filepaths
        self.box_size = box_size
        self.color_model = color_model
        super(AddCroppedBackground, self).__init__()

    def _crop_image(self, background_image):
        H, W = background_image.shape[:2]
        if (self.box_size >= H) or (self.box_size >= W):
            return None
        x_min = np.random.randint(0, W - self.box_size)
        y_min = np.random.randint(0, H - self.box_size)
        x_max = int(x_min + self.box_size)
        y_max = int(y_min + self.box_size)
        if (y_min > y_max) or (x_min > x_max):
            return None
        else:
            cropped_image = background_image[y_min:y_max, x_min:x_max]
            return cropped_image

    def call(self, image):
        if image.shape[-1] != 4:
            raise ValueError('Provided image does not contain alpha mask.')
        H, W = image.shape[:2]
        image_shape = (H, W, 3)
        foreground, alpha = np.split(image, [3], -1)
        alpha = alpha / 255
        random_arg = random.randint(0, len(self.image_filepaths))
        background = ops.load_image(self.image_filepaths[random_arg])
        if self.color_model == 'RGB':
            background = ops.convert_image(background, ops.BGR2RGB)
        background = self._crop_image(background)
        if background is None:
            background = np.ones((image_shape)) * random.randint(0, 256, 3)
        background = (1.0 - alpha) * background.astype(float)
        foreground = alpha * foreground
        image = foreground + background
        return image


class PixelBlur(Processor):
    """Add blur to image by downscaling and upscaling back to original shape.
    # Arguments:
        probability: Float, indication the probability of successfully
            applying this transformation.
    """
    def __init__(self, probability=.5, scale=0.25):
        self.scale = scale
        super(PixelBlur, self).__init__(probability)

    def call(self, image):
        H, W = image.shape[:2]
        reduced_H = int(self.scale * H)
        reduced_W = int(self.scale * W)
        image = ops.resize_image(image, (reduced_H, reduced_W))
        image = ops.resize_image(image, (H, W))
        return image


class GaussianBlur(Processor):
    """Apply Gaussian blur to image.
    # Arguments:
        probability: Float, indication the probability of successfully
            applying this transformation.
    """
    def __init__(self, probability=.5):
        super(GaussianBlur, self).__init__(probability)

    def call(self, image):
        H, W = image.shape[:2]
        image = ops.gaussian_blur(image, (5, 5))
        return image


class MedianBlur(Processor):
    """Apply median blur to image.
    # Arguments:
        probability: Float, indication the probability of successfully
            applying this transformation.
    """
    def __init__(self, probability=.5):
        super(MedianBlur, self).__init__(probability)

    def call(self, image):
        image = ops.median_blur(image, 5)
        return image


class RandomBlur(Processor):
    """Apply a randomly selected blur to image. Possible blurs are:
            PixelBlur, GaussianBlur and MedianBlur.
    # Arguments:
        probability: Float, indication the probability of successfully
            applying this transformation.
    """
    def __init__(self, probability=.5):
        self.blurs = [GaussianBlur(probability),
                      MedianBlur(probability),
                      PixelBlur(probability)]
        super(RandomBlur, self).__init__()

    def call(self, image):
        blur = np.random.choice(self.blurs)
        image = blur(image)
        return image


class ResizeImage(Processor):
    """Resize image.
    # Arguments
        size: Integer indicating the new shape (size, size) of the image.
    """
    def __init__(self, shape):
        self.shape = shape
        super(ResizeImage, self).__init__()

    def call(self, image):
        image = ops.resize_image(image, self.shape)
        return image


class ResizeImages(Processor):
    """ Resize cropped images
    # Arguments
        shape: List of two integers indicating the new shape of the image.
    """
    def __init__(self, shape):
        self.shape = shape
        super(ResizeImages, self).__init__()

    def call(self, images):
        images = [ops.resize_image(image, self.shape) for image in images]
        return images


class RandomSaturation(Processor):
    """Applies random saturation to an image in HSV space.
    # Arguments
        lower: Float, indicating the lower bound of the random number
            to be multiplied with the HSV image.
        upper: Float, indicating the upper bound of the random number
        to be multiplied with the HSV image.
    """
    def __init__(self, lower=0.3, upper=1.5, probability=0.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        super(RandomSaturation, self).__init__(probability)

    def call(self, image):
        image[:, :, 1] *= random.uniform(self.lower, self.upper)
        image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
        return image


class RandomHue(Processor):
    """Applies random hue to an image in HSV space.
    # Arguments
        delta: Integer, indicating the range (-delta, delta ) of possible
            hue values.
    """
    def __init__(self, delta=18, probability=0.5):
        assert delta >= 0.0 and delta <= 179.0
        self.delta = delta
        super(RandomHue, self).__init__(probability)

    def call(self, image):
        image[:, :, 0] += random.uniform(-self.delta, self.delta)
        image[:, :, 0][image[:, :, 0] > 179.0] -= 179.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 179.0
        return image


class RandomLightingNoise(Processor):
    """Applies random lighting noise by swapping channel colors.
    """
    def __init__(self, probability=0.5):
        self.permutations = ((0, 1, 2), (0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))
        super(RandomLightingNoise, self).__init__(probability)

    def call(self, image):
        swap = self.permutations[random.randint(len(self.permutations))]
        image = image[:, :, swap]
        return image


class ConvertColor(Processor):
    """Converts image to a different color space.
    # Arguments
        flag: Flag found in ``ops``indicating transform e.g. ops.BGR2RGB
    """
    def __init__(self, flag):
        self.flag = flag
        super(ConvertColor, self).__init__()

    def call(self, image):
        image = ops.convert_image(image, self.flag)
        return image


class RandomBrightness(Processor):
    """Add random brightness by adding a random constant to all image values.
            hue values.
    # Arguments:
        delta: Integer, indicating the range (-delta, delta ) of possible
            constant values to be added.
        probability: Float, indication the probability of successfully
            applying this transformation.
    """
    def __init__(self, delta=32, probability=0.5):

        if not (0.0 <= delta <= 255.0):
            raise ValueError("'delta' has to be between 0 and 255")
        if not (0.0 <= probability <= 1):
            raise ValueError("'probability' has to be between 0 and 1")

        self.delta = delta
        super(RandomBrightness, self).__init__(probability)

    def call(self, image):
        random_brightness = np.random.uniform(-self.delta, self.delta)
        image = image + random_brightness
        image = np.clip(image, 0, 255)
        return image


class RandomContrast(Processor):
    """Applies random contrast to an image in BGR/RGB space by multiplying all
        image values with a random number.
    # Arguments
        lower: Float, indicating the lower bound of the random number
            to be multiplied with the BGR/RGB image.
        upper: Float, indicating the upper bound of the random number
        to be multiplied with the BGR/RGB image.
    """
    def __init__(self, lower=0.5, upper=1.5, probability=0.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        super(RandomContrast, self).__init__(probability)

    # expects float image
    def call(self, image):
        alpha = random.uniform(self.lower, self.upper)
        image = image * alpha
        image = np.clip(image, 0, 255)
        return image


class RandomImageCrop(Processor):
    """Crops and returns random patch of an image.
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        super(RandomImageCrop, self).__init__()

    def call(self, image):
        height, width = image.shape[:2]
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])

                # cut the crop from the image
                current_image = current_image[
                    rect[1]:rect[3], rect[0]:rect[2], :]
                return image


class ExpandImage(Processor):
    """Expand image size up to 2x, 3x, 4x and fill values with mean color.
    This transformation is applied with a probability of 50%.
    # Arguments
        mean: List indicating BGR/RGB channel-wise mean color.
    """
    def __init__(self, mean, probability=0.5):
        self.mean = mean
        super(ExpandImage, self).__init__(probability)

    def call(self, image):
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expanded_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expanded_image[:, :, :] = self.mean
        expanded_image[int(top):int(top + height),
                       int(left):int(left + width)] = image
        return expanded_image


class AddOcclusion(Processor):
    """ Adds occlusion to image
    # Arguments
        max_radius_scale: Maximum radius in scale with respect to image i.e.
            each vertex radius from the polygon is sampled
            from [0, max_radius_scale]. This radius is later multiplied by
            the image dimensions.
    """
    def __init__(self, max_radius_scale=.5, probability=.5):
        super(AddOcclusion, self).__init__(probability)
        self.max_radius_scale = max_radius_scale

    def call(self, image):
        height, width = image.shape[:2]
        max_distance = np.max((height, width)) * self.max_radius_scale
        num_vertices = np.random.randint(3, 7)
        angle_between_vertices = 2 * np.pi / num_vertices
        initial_angle = np.random.uniform(0, 2 * np.pi)
        center = np.random.rand(2) * np.array([width, height])
        vertices = np.zeros((num_vertices, 2), dtype=np.int32)
        for vertex_arg in range(num_vertices):
            angle = initial_angle + (vertex_arg * angle_between_vertices)
            vertex = np.array([np.cos(angle), np.sin(angle)])
            vertex = np.random.uniform(0, max_distance) * vertex
            vertices[vertex_arg] = (vertex + center).astype(np.int32)
        color = np.random.randint(0, 256, 3).tolist()
        ops.draw_filled_polygon(image, vertices, color)
        return image


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
        return ops.show_image(image, self.window_name, self.wait)
