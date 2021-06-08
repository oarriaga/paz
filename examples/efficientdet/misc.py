import h5py
import numpy as np
import tensorflow as tf
import cv2

# Mock input image.
# raw_images = tf.random.uniform((1, 512, 512, 3),
#                                      dtype=tf.dtypes.float32,
#                                      seed=1) * 255
raw_images = cv2.imread('/home/deepan/Downloads/000000290320.jpg')
# raw_images = cv2.imread('./img.png')
raw_images = raw_images[np.newaxis]
raw_images = tf.convert_to_tensor(raw_images, dtype=tf.dtypes.float32)


def read_hdf5(path):
    """A function to read weights from h5 file."""
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f:
        f.visit(keys.append)
        for key in keys:
            if ':' in key:
                weights[f[key].name] = f[key][()]
    return weights


def load_pretrained_weights(model, weight_file_path):
    """
    A self-made manual method to copy weights from
    the official EfficientDet to this implementation.
    """
    pretrained_weights = read_hdf5(weight_file_path)
    assert len(model.weights) == len(pretrained_weights)
    str_appender = ['efficientnet-b0/',
                    'resample_p6/',
                    'fpn_cells/',
                    'class_net/',
                    'box_net/']
    for n, i in enumerate(model.weights):
        name = i.name
        for appenders in str_appender:
            if appenders in name:
                name = '/' + appenders + name
        if 'batch_normalization' in name:
            name = name.replace('batch_normalization',
                                'tpu_batch_normalization')
        if name in list(pretrained_weights.keys()):
            if model.weights[n].shape == pretrained_weights[name].shape:
                model.weights[n].assign(pretrained_weights[name])
            else:
                ValueError('Shape mismatch for weights of same name.')
        else:
            ValueError("Weight with %s not found." % name)
    return model


def preprocess_images(image, image_size):
    mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]  # imagenet rgb mean
    std_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]  # imagenet rgb std
    image = tf.cast(image, dtype=tf.float32)
    image -= tf.constant(mean_rgb, shape=(1, 1, 3), dtype=tf.float32)
    image /= tf.constant(std_rgb, shape=(1, 1, 3), dtype=tf.float32)

    crop_offset_y = tf.constant(0)
    crop_offset_x = tf.constant(0)
    height = tf.cast(tf.shape(image)[1], tf.float32)
    width = tf.cast(tf.shape(image)[2], tf.float32)
    image_scale_y = tf.cast(image_size[1], tf.float32) / height
    image_scale_x = tf.cast(image_size[2], tf.float32) / width
    image_scale = tf.minimum(image_scale_x, image_scale_y)

    scaled_height = tf.cast(height * image_scale, tf.int32)
    scaled_width = tf.cast(width * image_scale, tf.int32)
    scaled_image = tf.image.resize(image,
                                   [scaled_height, scaled_width],
                                   method=tf.image.ResizeMethod.BILINEAR)
    scaled_image = scaled_image[
                   :,
                   crop_offset_y: crop_offset_y + image_size[1],
                   crop_offset_x: crop_offset_x + image_size[2],
                   :]
    output_image = tf.image.pad_to_bounding_box(scaled_image,
                                                0,
                                                0,
                                                image_size[1],
                                                image_size[2])
    image = tf.cast(output_image, tf.float32)
    image_scale = 1 / image_scale
    return image, image_scale
