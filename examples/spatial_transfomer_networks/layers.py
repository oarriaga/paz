from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    # References
        [1]  Spatial Transformer Networks, Max Jaderberg, et al.
        [2]  https://github.com/skaae/transformer_network
        [3]  https://github.com/EderSantana/seya
        [4]  https://github.com/oarriaga/STN.keras
    """

    def __init__(self, output_size, dynamic=True, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(dynamic=dynamic, **kwargs)

    def get_config(self):
        return {'output_size': self.output_size}

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        image, affine_transforms = tensors
        batch_size, num_channels = K.shape(image)[0], K.shape(image)[3]
        affine_transforms = K.reshape(affine_transforms, (batch_size, 2, 3))
        grids = self._make_a_grid_per_batch(*self.output_size, batch_size)
        grids = K.batch_dot(affine_transforms, grids)
        interpolated_image = self._interpolate(image, grids, self.output_size)
        new_shape = (batch_size, *self.output_size, num_channels)
        interpolated_image = K.reshape(interpolated_image, new_shape)
        return interpolated_image

    def _make_grid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)
        return grid

    def _make_a_grid_per_batch(self, height, width, batch_size):
        grid = self._make_grid(height, width)
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _interpolate(self, image, grids, output_size):
        batch_size, height, width, num_channels = K.shape(image)
        x = K.cast(K.flatten(grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(grids[:, 1:2, :]), dtype='float32')
        x, y = self._to_image_coordinates(x, y, (height, width))
        x_min, y_min, x_max, y_max = self._compute_corners(x, y)
        x_min, y_min = self._clip_to_valid_coordinates((x_min, y_min), image)
        x_max, y_max = self._clip_to_valid_coordinates((x_max, y_max), image)
        offsets = self._compute_offsets_for_flat_batch(image, output_size)
        indices = self._calculate_indices(
            offsets, (x_min, y_min), (x_max, y_max), width)
        flat_images = K.reshape(image, shape=(-1, num_channels))
        flat_images = K.cast(flat_images, dtype='float32')
        pixel_values = self._gather_pixel_values(flat_images, indices)
        x_min, y_min = self._cast_points_to_float((x_min, y_min))
        x_max, y_max = self._cast_points_to_float((x_max, y_max))
        areas = self._calculate_areas(x, y, (x_min, y_min), (x_max, y_max))
        return self._compute_interpolations(areas, pixel_values)

    def _to_image_coordinates(self, x, y, shape):
        x = (0.5 * (x + 1.0)) * K.cast(shape[1], dtype='float32')
        y = (0.5 * (y + 1.0)) * K.cast(shape[0], dtype='float32')
        return x, y

    def _compute_corners(self, x, y):
        x_min, y_min = K.cast(x, 'int32'), K.cast(y, 'int32')
        x_max, y_max = x_min + 1, y_min + 1
        return x_min, y_min, x_max, y_max

    def _clip_to_valid_coordinates(self, points, image):
        x, y = points
        max_y = K.int_shape(image)[1] - 1
        max_x = K.int_shape(image)[2] - 1
        x = K.clip(x, 0, max_x)
        y = K.clip(y, 0, max_y)
        return x, y

    def _compute_offsets_for_flat_batch(self, image, output_size):
        batch_size, height, width = K.shape(image)[0:3]
        coordinates_per_batch = K.arange(0, batch_size) * (height * width)
        coordinates_per_batch = K.expand_dims(coordinates_per_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        coordinates_per_batch_per_pixel = K.repeat_elements(
            coordinates_per_batch, flat_output_size, axis=1)
        return K.flatten(coordinates_per_batch_per_pixel)

    def _calculate_indices(
            self, base, top_left_corners, bottom_right_corners, width):
        (x_min, y_min), (x_max, y_max) = top_left_corners, bottom_right_corners
        y_min_offset = base + (y_min * width)
        y_max_offset = base + (y_max * width)
        indices_top_left = y_min_offset + x_min
        indices_top_right = y_max_offset + x_min
        indices_low_left = y_min_offset + x_max
        indices_low_right = y_max_offset + x_max
        return (indices_top_left, indices_top_right,
                indices_low_left, indices_low_right)

    def _gather_pixel_values(self, flat_image, indices):
        pixel_values_A = K.gather(flat_image, indices[0])
        pixel_values_B = K.gather(flat_image, indices[1])
        pixel_values_C = K.gather(flat_image, indices[2])
        pixel_values_D = K.gather(flat_image, indices[3])
        return (pixel_values_A, pixel_values_B, pixel_values_C, pixel_values_D)

    def _calculate_areas(self, x, y, top_left_corners, bottom_right_corners):
        (x_min, y_min), (x_max, y_max) = top_left_corners, bottom_right_corners
        area_A = K.expand_dims(((x_max - x) * (y_max - y)), 1)
        area_B = K.expand_dims(((x_max - x) * (y - y_min)), 1)
        area_C = K.expand_dims(((x - x_min) * (y_max - y)), 1)
        area_D = K.expand_dims(((x - x_min) * (y - y_min)), 1)
        return area_A, area_B, area_C, area_D

    def _cast_points_to_float(self, points):
        return K.cast(points[0], 'float32'), K.cast(points[1], 'float32')

    def _compute_interpolations(self, areas, pixel_values):
        weighted_area_A = pixel_values[0] * areas[0]
        weighted_area_B = pixel_values[1] * areas[1]
        weighted_area_C = pixel_values[2] * areas[2]
        weighted_area_D = pixel_values[3] * areas[3]
        interpolation = (weighted_area_A + weighted_area_B +
                         weighted_area_C + weighted_area_D)
        return interpolation
