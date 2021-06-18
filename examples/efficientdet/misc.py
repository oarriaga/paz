import os
import h5py
import numpy as np
import tensorflow as tf
from PIL import Image

# Mock input image.
path_to_paz = '/media/deepan/externaldrive1/Gini/project_repos/'
directory_path = 'paz/examples/efficientdet/'
file_name = 'img2.png'
file_path = path_to_paz + directory_path + file_name
raw_images = Image.open(file_path)
raw_images = np.asarray(raw_images)
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
                weights[f[key].name] = \
                    f[key][()]
    return weights


def load_pretrained_weights(model,
                            weight_file_path,
                            using_renamed_layers_from_hdf5=False):
    """
    A self-made manual method to copy weights from
    the official EfficientDet to this implementation.
    """
    pretrained_weights = read_hdf5(weight_file_path)
    assert len(model.weights) == len(pretrained_weights)
    string_appender = ['efficientnet-b0/',
                       'resample_p6/',
                       'fpn_cells/',
                       'class_net/',
                       'box_net/']
    for n, i in enumerate(model.weights):
        name = i.name
        if using_renamed_layers_from_hdf5:
            name = '/' + name
        else:
            for appenders in string_appender:
                if appenders in name:
                    name = '/' + appenders + name
            if 'batch_normalization' in name:
                name = name.replace('batch_normalization',
                                    'tpu_batch_normalization')
        if name in list(pretrained_weights.keys()):
            if model.weights[n].shape == pretrained_weights[name].shape:
                model.weights[n].assign(pretrained_weights[name])
            else:
                raise ValueError('Shape mismatch for weights of same name.')
        else:
            print('not copying due to no name', name)
            raise ValueError("Weight with %s not found." % name)
    return model


def create_renamed_hdf5(path):
    """A function to update the weights name of a h5 file.

    Arguments:
        path: Path of the hdf5 weight file [COPY].
        This function updates layer names of the
        passed hdf5 file directly. Therefore, make a copy.
    """
    keys = []
    # Weight name substring to be searched
    search_string = ['/box_net/box_net/',
                     '/class_net/class_net/',
                     '/efficientnet-b0/efficientnet-b0/',
                     '/fpn_cells/fpn_cells/',
                     '/resample_p6/resample_p6/',
                     'tpu_batch_normalization']
    # Weight name to replace the above searched substring
    replace_string = ['box_net/',
                      'class_net/',
                      'efficientnet-b0/',
                      'fpn_cells/',
                      'resample_p6/',
                      'batch_normalization']
    with h5py.File(path, 'r+') as f:
        f.visit(keys.append)
        for i in range(len(search_string)):
            for key_args in range(len(keys)):
                keys = []
                f.visit(keys.append)
                if search_string[i] in f[keys[key_args]].name:
                    new_key = f[keys[key_args]].name.replace(search_string[i],
                                                             replace_string[i])
                    f[new_key] = f[keys[key_args]]
                    del f[keys[key_args]]
    f.close()


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
    path_to_paz = '/media/deepan/externaldrive1/Gini/project_repos/'
    directory_path = 'paz/examples/efficientdet/'
    save_file_path = os.path.join(path_to_paz + directory_path + file_name)
    Image.fromarray(image.astype('uint8')).save(save_file_path)
    print('writing file to %s' % save_file_path)
