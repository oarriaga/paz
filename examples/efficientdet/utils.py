import os
import cv2
import numpy as np
import tensorflow as tf
import anchors
from paz.backend.boxes import to_center_form


# Mock input image.
path_to_paz = '/media/deepan/externaldrive1/project_repos/'
directory_path = 'paz_versions/paz_efficientdet/examples/efficientdet/'
file_name = 'img2.png'
file_path = path_to_paz + directory_path + file_name
raw_images = cv2.imread(file_path)
raw_images = cv2.cvtColor(raw_images, cv2.COLOR_BGR2RGB)
raw_images = raw_images[np.newaxis]
raw_images = tf.convert_to_tensor(raw_images, dtype=tf.dtypes.float32)


def get_activation(features, activation):
    """Apply non-linear activation function to features provided.

    # Arguments
        features: Tensor, representing an input feature map
        to be pass through an activation function.
        activation: A string specifying the activation function
        type.

    # Returns
        activation function: features transformed by the
        activation function.
    """
    if activation in ('silu', 'swish'):
        return tf.nn.swish(features)
    elif activation == 'relu':
        return tf.nn.relu(features)
    else:
        raise ValueError('Unsupported activation fn {}'.format(activation))


def get_drop_connect(features, is_training, survival_rate):
    """Drop the entire conv with given survival probability.
    Deep Networks with Stochastic Depth, https://arxiv.org/pdf/1603.09382.pdf

    # Arguments
        features: Tensor, input feature map to undergo
        drop connection.
        is_training: Bool specifying the training phase.
        survival_rate: Float, survival probability to drop
        input convolution features.

    # Returns
        output: Tensor, output feature map after drop connect.
    """
    if not is_training:
        return features
    batch_size = tf.shape(features)[0]
    random_tensor = survival_rate
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1],
                                       dtype=features.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = features / survival_rate * binary_tensor
    return output


def get_prior_boxes(min_level, max_level, num_scales, aspect_ratios,
                    anchor_scale, image_size):
    """
    Function to generate prior boxes.

    # Arguments
    min_level: Int, minimum level for features.
    max_level: Int, maximum level for features.
    num_scales: Int, specifying the number of scales in the anchor boxes.
    aspect_ratios: List, specifying the aspect ratio of the
    default anchor boxes. Computed with k-mean on COCO dataset.
    num_classes: Int, specifying the number of class in the output.
    image_size: Int, size of the input image.

    # Returns
    prior_boxes: Tensor, Prior anchor boxes corresponding to the
    feature map size of each feature level.
    """
    prior_anchors = anchors.Anchors(min_level, max_level, num_scales,
                                    aspect_ratios, anchor_scale, image_size)
    prior_boxes = prior_anchors.boxes
    prior_boxes = tf.expand_dims(prior_boxes, axis=0)
    s1, s2, s3, s4 = tf.split(prior_boxes, num_or_size_splits=4, axis=2)
    prior_boxes = tf.concat([s2, s1, s4, s3], axis=2)
    prior_boxes = prior_boxes[0]
    prior_boxes = to_center_form(prior_boxes)
    return prior_boxes


def merge_level_outputs(class_outputs, box_outputs, num_levels, num_classes):
    """
    Merges all feature levels into single tensor.

    # Arguments
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.

    # Returns
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates irrespective of feature
        levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes irrespective of feature levels.
    """
    class_outputs_all, box_outputs_all = [], []
    batch_size = tf.shape(class_outputs[0])[0]
    for level in range(0, num_levels):
        class_outputs_all.append(tf.reshape(
            class_outputs[level], [batch_size, -1, num_classes]))
        box_outputs_all.append(tf.reshape(
            box_outputs[level], [batch_size, -1, 4]))
    return tf.concat(class_outputs_all, 1), tf.concat(box_outputs_all, 1)


def process_outputs(class_outputs, box_outputs, num_levels, num_classes):
    """
    Merges all feature levels into single tensor and combines box offsets
    and class scores.

    # Arguments
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.
        num_levels: Int, number of levels considered at efficientnet features.
        num_classes: Int, number of classes in the dataset.

    # Returns
        outputs: Tensor, returned only when the return_base flag is false.
        Processed outputs by merging the features at all levels. Each row
        corresponds to box coordinate offsets and sigmoid of the class logits.
    """
    class_outputs, box_outputs = merge_level_outputs(
        class_outputs, box_outputs, num_levels, num_classes)
    s1, s2, s3, s4 = tf.split(box_outputs, num_or_size_splits=4, axis=2)
    box_outputs = tf.concat([s2, s1, s4, s3], axis=2)
    cls_outputs = tf.sigmoid(class_outputs)
    outputs = tf.concat([box_outputs, cls_outputs], axis=2)
    return outputs


def preprocess_images(image, image_size):
    """
    Preprocess image for EfficientDet model.

    # Arguments
        image: Tensor, raw input image to be preprocessed
        of shape [bs, h, w, c]
        image_size: Tensor, size to resize the raw image
        of shape [bs, new_h, new_w, c]

    # Returns
        image: Tensor, resized and preprocessed image
        image_scale: Tensor, scale to reconstruct each of
        the raw images to original size from the resized
        image.
    """
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
    scaled_image = tf.image.resize(image, [scaled_height, scaled_width],
                                   method=tf.image.ResizeMethod.BILINEAR)
    scaled_image = scaled_image[
                   :,
                   crop_offset_y: crop_offset_y + image_size[1],
                   crop_offset_x: crop_offset_x + image_size[2],
                   :]
    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, image_size[1], image_size[2])
    image = tf.cast(output_image, tf.float32)
    image_scale = 1 / image_scale
    return image, image_scale


def save_file(file_name, image):
    path_to_paz = '/media/deepan/externaldrive1/project_repos/'
    directory_path = 'paz_versions/paz_efficientdet/examples/efficientdet/'
    save_file_path = os.path.join(path_to_paz + directory_path + file_name)
    cv2.imwrite(save_file_path,  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print('writing file to %s' % save_file_path)
