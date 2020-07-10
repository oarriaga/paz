from paz.abstract import Processor
from paz.backend.image import draw_circle
from paz.backend.image.draw import GREEN
import numpy as np
import cv2


def warp_affine(image, matrix, fill_color=[0, 0, 0]):
    """ Transforms `image` using an affine `matrix` transformation.

    # Arguments
        image: Numpy array.
        matrix: Numpy array of shape (2,3) indicating affine transformation.
        fill_color: List of three floats representing a color.
    """
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill_color)


def translate_image(image, translation, fill_color):
    """Translate image.

    # Arguments
        image: Numpy array.
        translation: A list of length two indicating the x,y translation values
        fill_color: List of three floats representing a color.

    # Returns
        Numpy array
    """
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0], matrix[1, 1] = 1.0, 1.0
    matrix[0, 2], matrix[1, 2] = translation
    image = warp_affine(image, matrix, fill_color)
    return image


def translate_keypoints(keypoints, translation):
    """Translate keypoints.

    # Arguments
        kepoints: Numpy array of shape ``(num_keypoints, 2)``.
        translation: A list of length two indicating the x,y translation values

    # Returns
        Numpy array
    """
    return keypoints + translation


class TranslateImage(Processor):
    """Applies a translation of image.
    The translation is a list of length two indicating the x, y values.
    # Arguments
        fill_color: List of three integers indicating the
            color values e.g. [0,0,0]
    """
    def __init__(self, fill_color=None):
        super(TranslateImage, self).__init__()
        self.fill_color = fill_color

    def call(self, image, translation):
        return translate_image(image, translation, self.fill_color)


class TranslateKeypoints(Processor):
    """Applies a translation to keypoints.
    The translation is a list of length two indicating the x, y values.
    """
    def __init__(self):
        super(TranslateKeypoints, self).__init__()

    def call(self, keypoints, translation):
        return translate_keypoints(keypoints, translation)


def sample_scaled_translation(delta_scale, image_shape):
    """Samples a scaled translation from a uniform distribution.

    # Arguments
        delta_scale: List with two elements having the normalized deltas.
            e.g. ''[.25, .25]''.
        image_shape: List containing the height and width of the image.
    """
    x_delta_scale, y_delta_scale = delta_scale
    x = image_shape[1] * np.random.uniform(-x_delta_scale, x_delta_scale)
    y = image_shape[0] * np.random.uniform(-y_delta_scale, y_delta_scale)
    return [x, y]


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
    def __init__(self, rotation_range=30, fill_color=None, **kwargs):
        super(RandomKeypointRotation, self).__init__(**kwargs)
        self.rotation_range = rotation_range
        self.fill_color = fill_color

    def _calculate_image_center(self, image):
        return (int(image.shape[0] / 2), int(image.shape[1] / 2))

    def _rotate_image(self, image, degrees):
        center = self._calculate_image_center(image)
        matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
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

    def call(self, image, keypoints):
        degrees = np.random.uniform(-self.rotation_range, self.rotation_range)
        image = self._rotate_image(image, degrees)
        image_center = self._calculate_image_center(image)
        radians = self._degrees_to_radians(degrees)
        keypoints = self._rotate_keypoints(keypoints, radians, image_center)
        return image, keypoints


def draw_circles(image, points, color=GREEN, radius=3):
    for point in points:
        draw_circle(image, point, color, radius)
    return image


if __name__ == '__main__':
    from facial_keypoints import FacialKeypoints
    from paz.backend.image import show_image

    data_manager = FacialKeypoints('dataset/', 'train')
    faces, keypoints = data_manager.load_data()
    image, keypoints_set = faces[0], keypoints[0]
    # image = draw_circles(original_image.copy(), keypoints_set.astype('int'))
    # show_image(image.astype('uint8'))
    rotate_keypoints = RandomKeypointRotation(30)
    random_translate = RandomKeypointTranslation(probability=1)
    for arg in range(100):
        original_image, kp = image.copy(), keypoints_set.copy()
        original_image, kp = rotate_keypoints(original_image, kp)
        original_image, kp = random_translate(original_image, kp)
        original_image = draw_circles(original_image, kp.astype('int'))
        show_image(original_image.astype('uint8'))
