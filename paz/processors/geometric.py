import numpy as np

from ..abstract import Processor

from ..backend.boxes import flip_left_right
from ..backend.boxes import to_image_coordinates
from ..backend.boxes import to_normalized_coordinates
from ..backend.boxes import compute_iou
from ..backend.image import warp_affine
from ..backend.image import translate_image
from ..backend.image import sample_scaled_translation
from ..backend.image import get_rotation_matrix
from ..backend.image import calculate_image_center
from ..backend.image import get_affine_transform
from ..backend.keypoints import translate_keypoints
from ..backend.keypoints import rotate_point2D
from ..backend.standard import resize_with_same_aspect_ratio
from ..backend.standard import get_transformation_scale


class RandomFlipBoxesLeftRight(Processor):
    """Flips image and implemented labels horizontally.
    """
    def __init__(self):
        super(RandomFlipBoxesLeftRight, self).__init__()

    def call(self, image, boxes):
        if np.random.randint(0, 2):
            boxes = flip_left_right(boxes, image.shape[1])
            image = image[:, ::-1]
        return image, boxes


class ToImageBoxCoordinates(Processor):
    """Convert normalized box coordinates to image-size box coordinates.
    """
    def __init__(self):
        super(ToImageBoxCoordinates, self).__init__()

    def call(self, image, boxes):
        boxes = to_image_coordinates(boxes, image)
        return image, boxes


class ToNormalizedBoxCoordinates(Processor):
    """Convert image-size box coordinates to normalized box coordinates.
    """
    def __init__(self):
        super(ToNormalizedBoxCoordinates, self).__init__()

    def call(self, image, boxes):
        boxes = to_normalized_coordinates(boxes, image)
        return image, boxes


class RandomSampleCrop(Processor):
    """Crops and image while adjusting the bounding boxes.
    Boxes should be in point form.
    # Arguments
        probability: Float between ''[0, 1]''.
    """
    def __init__(self, probability=0.5):
        self.probability = probability
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
        super(RandomSampleCrop, self).__init__()

    def call(self, image, boxes):
        if self.probability < np.random.rand():
            return image, boxes
        labels = boxes[:, -1:]
        boxes = boxes[:, :4]
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                boxes = np.hstack([boxes, labels])
                return image, boxes

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = compute_iou(rect, boxes)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                return current_image, np.hstack(
                    [current_boxes, current_labels])


class Expand(Processor):
    """Expand image size up to 2x, 3x, 4x and fill values with mean color.
    This transformation is applied with a probability of 50%.

    # Arguments
        max_ratio: Float.
        mean: None/List: If `None` expanded image is filled with
            the image mean.
        probability: Float between ''[0, 1]''.
    """
    def __init__(self, max_ratio=2, mean=None, probability=0.5):
        super(Expand, self).__init__()
        self.max_ratio = max_ratio
        self.mean = mean
        self.probability = probability

    def call(self, image, boxes):
        if self.probability < np.random.rand():
            return image, boxes
        height, width, num_channels = image.shape
        ratio = np.random.uniform(1, self.max_ratio)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)
        expanded_image = np.zeros((int(height * ratio),
                                   int(width * ratio), num_channels),
                                  dtype=image.dtype)

        if self.mean is None:
            expanded_image[:, :, :] = np.mean(image, axis=(0, 1))
        else:
            expanded_image[:, :, :] = self.mean

        expanded_image[int(top):int(top + height),
                       int(left):int(left + width)] = image
        expanded_boxes = boxes.copy()
        expanded_boxes[:, 0:2] = boxes[:, 0:2] + (int(left), int(top))
        expanded_boxes[:, 2:4] = boxes[:, 2:4] + (int(left), int(top))
        return expanded_image, expanded_boxes


class ApplyTranslation(Processor):
    """Applies a translation of image and labels.

    # Arguments
        translation: A list of length two indicating the x,y translation values
        fill_color: List of three integers indicating the
            color values e.g. ''[0, 0, 0]''
    """
    def __init__(self, translation, fill_color=None):
        super(ApplyTranslation, self).__init__()
        self._matrix = np.zeros((2, 3), dtype=np.float32)
        self._matrix[0, 0], self._matrix[1, 1] = 1.0, 1.0
        self.fill_color = fill_color
        self.translation = translation

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):
        if translation is None:
            self._translation = None
        elif len(translation) == 2:
            self._translation = translation
            self._matrix[0, 2], self._matrix[1, 2] = translation
        else:
            raise ValueError('Translation should be `None` or have length two')

    def call(self, image, keypoints=None):
        height, width = image.shape[:2]
        if self.fill_color is None:
            fill_color = np.mean(image, axis=(0, 1))
        image = warp_affine(image, self._matrix, fill_color)
        if keypoints is not None:
            keypoints[:, 0] = keypoints[:, 0] + self.translation[0]
            keypoints[:, 1] = keypoints[:, 1] + self.translation[1]
            return image, keypoints
        return image


class RandomTranslation(Processor):
    """Applies a random translation to image and labels

    # Arguments
        delta_scale: List with two elements having the normalized deltas.
            e.g. ''[.25, .25]''.

        fill_color: List of three integers indicating the
            color values e.g. ''[0, 0, 0]''.
    """
    def __init__(
            self, delta_scale=[0.25, 0.25], fill_color=None):
        super(RandomTranslation, self).__init__()
        self.delta_scale = delta_scale
        self.apply_translation = ApplyTranslation(None, fill_color)

    @property
    def delta_scale(self):
        return self._delta_scale

    @delta_scale.setter
    def delta_scale(self, delta_scale):
        x_delta_scale, y_delta_scale = delta_scale
        if (x_delta_scale < 0) or (y_delta_scale < 0):
            raise ValueError('Delta scale values should be a positive scalar')
        self._delta_scale = delta_scale

    def call(self, image):
        height, width = image.shape[:2]
        x_delta_scale, y_delta_scale = self.delta_scale
        x = image.shape[1] * np.random.uniform(-x_delta_scale, x_delta_scale)
        y = image.shape[0] * np.random.uniform(-y_delta_scale, y_delta_scale)
        self.apply_translation.translation = [x, y]
        return self.apply_translation(image)


class RandomKeypointTranslation(Processor):
    """Applies a random translation to image and keypoints.

    # Arguments
        delta_scale: List with two elements having the normalized deltas.
            e.g. ''[.25, .25]''.
        fill_color: ''None'' or List of three integers indicating the
            color values e.g. ''[0, 0, 0]''. If ''None'' mean channel values of
            the image will be calculated as fill values.
        probability: Float between ''[0, 1]''.
    """
    def __init__(self, delta_scale=[.2, .2], fill_color=None, probability=0.5):
        super(RandomKeypointTranslation, self).__init__()
        self.delta_scale = delta_scale
        self.fill_color = fill_color
        self.probability = probability

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        if not (0.0 < value <= 1.0):
            raise ValueError('Probability should be between "[0, 1]".')
        self._probability = value

    @property
    def delta_scale(self):
        return self._delta_scale

    @delta_scale.setter
    def delta_scale(self, delta_scale):
        x_delta_scale, y_delta_scale = delta_scale
        if (x_delta_scale < 0) or (y_delta_scale < 0):
            raise ValueError('Delta scale values should be positive')
        if (x_delta_scale > 1) or (y_delta_scale > 1):
            raise ValueError('Delta scale values should be less than one')
        self._delta_scale = delta_scale

    def _sample_random_translation(self, delta_scale, image_shape):
        x_delta_scale, y_delta_scale = delta_scale
        x = image_shape[1] * np.random.uniform(-x_delta_scale, x_delta_scale)
        y = image_shape[0] * np.random.uniform(-y_delta_scale, y_delta_scale)
        return [x, y]

    def call(self, image, keypoints):
        if self.probability >= np.random.rand():
            shape = image.shape[:2]
            translation = sample_scaled_translation(self.delta_scale, shape)
            if self.fill_color is None:
                fill_color = np.mean(image, axis=(0, 1))
            image = translate_image(image, translation, fill_color)
            keypoints = translate_keypoints(keypoints, translation)
        return image, keypoints


class RandomKeypointRotation(Processor):
    """Randomly rotate an images with its corresponding keypoints.

    # Arguments
        rotation_range: Int. indicating the max and min values in degrees
            of the uniform distribution ''[-range, range]'' from which the
            angles are sampled.
        fill_color: ''None'' or List of three integers indicating the
            color values e.g. ''[0, 0, 0]''. If ''None'' mean channel values of
            the image will be calculated as fill values.
    """
    def __init__(self, rotation_range=30, fill_color=None, probability=0.5):
        super(RandomKeypointRotation, self).__init__()
        self.rotation_range = rotation_range
        self.fill_color = fill_color
        self.probability = probability

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        if not (0.0 < value <= 1.0):
            raise ValueError('Probability should be between "[0, 1]".')
        self._probability = value

    def _calculate_image_center(self, image):
        return (int(image.shape[0] / 2), int(image.shape[1] / 2))

    def _rotate_image(self, image, degrees):
        center = self._calculate_image_center(image)
        matrix = get_rotation_matrix(center, degrees)
        if self.fill_color is None:
            fill_color = np.mean(image, axis=(0, 1))
        return warp_affine(image, matrix, fill_color)

    def _degrees_to_radians(self, degrees):
        # negative sign changes rotation direction to follow openCV convention.
        return - (3.14159 / 180) * degrees

    def _build_rotation_matrix(self, radians):
        return np.array([[np.cos(radians), - np.sin(radians)],
                         [np.sin(radians), + np.cos(radians)]])

    def _rotate_keypoints(self, keypoints, radians, image_center):
        keypoints = keypoints - image_center
        matrix = self._build_rotation_matrix(radians)
        keypoints = np.matmul(matrix, keypoints.T).T
        keypoints = keypoints + image_center
        return keypoints

    def _sample_rotation(self, rotation_range):
        return np.random.uniform(-rotation_range, rotation_range)

    def call(self, image, keypoints):
        if self.probability >= np.random.rand():
            degrees = self._sample_rotation(self.rotation_range)
            image = self._rotate_image(image, degrees)
            center = self._calculate_image_center(image)
            radians = self._degrees_to_radians(degrees)
            keypoints = self._rotate_keypoints(keypoints, radians, center)
        return image, keypoints


class RandomRotation(Processor):
    """Randomly rotate an images

    # Arguments
        rotation_range: Int. indicating the max and min values in degrees
            of the uniform distribution ``[-range, range]`` from which the
            angles are sampled.
        fill_color: ''None'' or List of three integers indicating the
            color values e.g. ``[0, 0, 0]``. If ``None`` mean channel values of
            the image will be calculated as fill values.
        probability: Float between 0 and 1.
    """
    def __init__(self, rotation_range=30, fill_color=None, probability=0.5):
        super(RandomRotation, self).__init__()
        self.rotation_range = rotation_range
        self.fill_color = fill_color
        self.probability = probability

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        if not (0.0 < value <= 1.0):
            raise ValueError('Probability should be between "[0, 1]".')
        self._probability = value

    def _calculate_image_center(self, image):
        return (int(image.shape[0] / 2), int(image.shape[1] / 2))

    def _rotate_image(self, image, degrees):
        center = self._calculate_image_center(image)
        matrix = get_rotation_matrix(center, degrees)
        if self.fill_color is None:
            fill_color = np.mean(image, axis=(0, 1))
        return warp_affine(image, matrix, fill_color)

    def _sample_rotation(self, rotation_range):
        return np.random.uniform(-rotation_range, rotation_range)

    def call(self, image):
        if self.probability >= np.random.rand():
            degrees = self._sample_rotation(self.rotation_range)
            image = self._rotate_image(image, degrees)
        return image


class TranslateImage(Processor):
    """Applies a translation of image.
    The translation is a list of length two indicating the x, y values.

    # Arguments
        fill_color: List of three integers indicating the
            color values e.g. ``[0, 0, 0]``
    """
    def __init__(self, fill_color=None):
        super(TranslateImage, self).__init__()
        self.fill_color = fill_color

    def call(self, image, translation):
        return translate_image(image, translation, self.fill_color)


class GetTransformationSize(Processor):
    """Calculate the transformation size for the imgae.
    The size is tuple of length two indicating the x, y values.

    # Arguments
        image: Numpy array
    """
    def __init__(self, input_size, multiple):
        super(GetTransformationSize, self).__init__()
        self.input_size = input_size
        self.multiple = multiple

    def call(self, image):
        size = resize_with_same_aspect_ratio(image, self.input_size,
                                             self.multiple)
        H, W = image.shape[:2]
        if W < H:
            size[0], size[1] = size[1], size[0]
        return size


class GetTransformationScale(Processor):
    """Calculate the transformation scale for the imgae.
    The scale is a numpy array of size two indicating the
    width and height scale.

    # Arguments
        image: Numpy array
        size: Numpy array of length 2
    """
    def __init__(self, scaling_factor):
        super(GetTransformationScale, self).__init__()
        self.scaling_factor = scaling_factor

    def call(self, image, size):
        scale = get_transformation_scale(image, size, self.scaling_factor)
        H, W = image.shape[:2]
        if W < H:
            scale[0], scale[1] = scale[1], scale[0]
        return scale


class GetSourceDestinationPoints(Processor):
    """Returns the source and destination points for affine transformation.

    # Arguments
        center: Numpy array of shape (2,). Center coordinates of image
        scale: Numpy array of shape (2,). Scale of width and height of image
        size: List of length 2. Size of image
    """
    def __init__(self, scaling_factor):
        super(GetSourceDestinationPoints, self).__init__()
        self.scaling_factor = scaling_factor

    def _calculate_third_point(self, point2D_a, point2D_b):
        difference = point2D_a - point2D_b
        return point2D_a + np.array([-difference[1],
                                     difference[0]], dtype=np.float32)

    def _get_transformation_source_point(self, scale, center):
        scale = scale * self.scaling_factor
        center_W = scale[0] / 2
        direction_vector = rotate_point2D([0, -center_W], 0)
        points = np.zeros((3, 2), dtype=np.float32)
        points[0, :] = center
        points[1, :] = center + direction_vector
        points[2:, :] = self._calculate_third_point(points[0, :], points[1, :])
        return points

    def _get_transformation_destination_point(self, output_size):
        center_W, center_H = np.array(output_size[:2]) / 2
        direction_vector = np.array([0, -center_W], np.float32)
        points = np.zeros((3, 2), dtype=np.float32)
        points[0, :] = [center_W, center_H]
        points[1, :] = np.array([center_W, center_H]) + direction_vector
        points[2:, :] = self._calculate_third_point(points[0, :], points[1, :])
        return points

    def call(self, center, scale, size):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])
        source_point = self._get_transformation_source_point(scale, center)
        destination_point = self._get_transformation_destination_point(size)
        return source_point, destination_point


class GetImageCenter(Processor):
    """Calculate the center of the image and add an offset to the center.

    # Arguments
        image: Numpy array
        offset: Float
    """
    def __init__(self, offset=0.5):
        super(GetImageCenter, self).__init__()
        self.offset = offset

    def _add_offset(self, x, offset):
        return (x + offset)

    def call(self, image):
        center_W, center_H = calculate_image_center(image)
        center_W = int(self._add_offset(center_W, self.offset))
        center_H = int(self._add_offset(center_H, self.offset))
        return np.array([center_W, center_H])


class WarpAffine(Processor):
    """Applies an affine transformation to an image

    # Arguments
        image: Numpy array
        transform: Numpy array. Transformation matrix
        size: Numpy array. Transformation size
    """
    def __init__(self):
        super(WarpAffine, self).__init__()

    def call(self, image, transform, size):
        image = warp_affine(image, transform, size=size)
        return image
