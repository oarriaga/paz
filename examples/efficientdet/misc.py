import os
import h5py
import numpy as np
import tensorflow as tf
from PIL import Image

# Mock input image.
path_to_paz = '/media/deepan/externaldrive1/project_repos/'
directory_path = 'paz/examples/efficientdet/'
file_name = 'img2.png'
file_path = path_to_paz + directory_path + file_name
raw_images = Image.open(file_path)
raw_images = np.asarray(raw_images)
raw_images = raw_images[np.newaxis]
raw_images = tf.convert_to_tensor(raw_images, dtype=tf.dtypes.float32)


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


def save_file(file_name, image):
    path_to_paz = '/media/deepan/externaldrive1/project_repos/'
    directory_path = 'paz/examples/efficientdet/'
    save_file_path = os.path.join(path_to_paz + directory_path + file_name)
    Image.fromarray(image.astype('uint8')).save(save_file_path)
    print('writing file to %s' % save_file_path)
