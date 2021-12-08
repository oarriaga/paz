import tensorflow as tf
from tensorflow.keras.layers import Layer
from backend_keypoints import find_max_location


class SegmentationDilation(Layer):
    def __init__(self, filter_size=21):
        super(SegmentationDilation, self).__init__()
        self.filter_size = filter_size
        filters = tf.ones((filter_size, filter_size, 1))
        self.kernel = filters / float(self.filter_size ** 2)

    def call(self, inputs):
        segmentation_map_height, segmentation_map_width, channels = inputs.shape
        scoremap_softmax = tf.nn.softmax(inputs)
        scoremap_foreground = tf.reduce_max(scoremap_softmax[:, :, 1:], -1)
        segmentationmap_foreground = tf.round(scoremap_foreground)
        max_loc = find_max_location(scoremap_foreground)

        sparse_indices = tf.reshape(max_loc, [1, 2])

        sparse_input = tf.SparseTensor(
            dense_shape=[segmentation_map_height, segmentation_map_width],
            values=[1.0], indices=sparse_indices)

        objectmap = tf.sparse.to_dense(sparse_input)
        num_passes = max(segmentation_map_height, segmentation_map_width) // (
                self.filter_size // 2)

        for pass_count in range(num_passes):
            objectmap = tf.reshape(objectmap, [1, segmentation_map_height,
                                               segmentation_map_width, 1])

            objectmap_dilated = tf.nn.dilation2d(
                input=objectmap, filters=self.kernel, strides=[1, 1, 1, 1],
                dilations=[1, 1, 1, 1], padding='SAME', data_format='NHWC')

            objectmap_dilated = tf.reshape(objectmap_dilated,
                                           [segmentation_map_height,
                                            segmentation_map_width])

            objectmap = tf.round(tf.multiply(segmentationmap_foreground,
                                             objectmap_dilated))

        objectmap = tf.reshape(objectmap, [segmentation_map_height,
                                           segmentation_map_width, 1])
        return objectmap.numpy()
