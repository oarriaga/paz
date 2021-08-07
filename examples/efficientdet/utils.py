import os
import cv2
import tensorflow as tf
import anchors
from paz.backend.boxes import to_center_form
import paz.processors as pr
from paz.abstract import SequentialProcessor
from paz.processors.image import LoadImage
from paz.processors.image import RGB_IMAGENET_MEAN, RGB_IMAGENET_STDEV


# Mock input image.
path_to_paz = '/media/deepan/externaldrive1/project_repos/'
directory_path = 'paz_versions/paz_efficientdet/examples/efficientdet/'
file_name = 'img2.png'
file_path = path_to_paz + directory_path + file_name
loader = LoadImage()
raw_images = loader(file_path)


class_names = ['person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', '0', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', '0', 'backpack', 'umbrella', '0',
               '0', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
               'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', '0', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', '0', 'dining table', '0', '0', 'toilet',
               '0', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', '0', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def get_class_name_efficientdet(dataset_name):
    if dataset_name == 'COCO':
        return class_names


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

    preprocessing = SequentialProcessor([
        pr.CastImage(float),
        pr.SubtractMeanImage(mean=RGB_IMAGENET_MEAN),
        pr.DivideStandardDeviationImage(standard_deviation=RGB_IMAGENET_STDEV),
        pr.ScaledResize(image_size=image_size),
        ])
    image, image_scale = preprocessing(image)
    return image, image_scale


def save_file(file_name, image):
    path_to_paz = '/media/deepan/externaldrive1/project_repos/'
    directory_path = 'paz_versions/paz_efficientdet/examples/efficientdet/'
    save_file_path = os.path.join(path_to_paz + directory_path + file_name)
    cv2.imwrite(save_file_path,  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print('writing file to %s' % save_file_path)
