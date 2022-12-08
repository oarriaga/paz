import tensorflow as tf
import tensorflow.keras.backend as K
from paz.processors.image import LoadImage
from tensorflow.keras.layers import Activation, Concatenate, Flatten, Reshape

# Mock input image.
file_name = ('/home/manummk95/Desktop/efficientdet_BKP/paz/'
             'examples/efficientdet/000132.jpg')
loader = LoadImage()
raw_images = loader(file_name)


def create_multibox_head(branch_tensors, num_levels, num_classes,
                         num_regressions=4):
    """Concatenates class and box outputs into single tensor.

    # Arguments:
        branch_tensors: List, containing efficientdet outputs.
        num_levels: Int, number of feature levels.
        num_classes: Int, number of output classes.
        num_regressions: Int, number of bounding box coordinate.

    # Returns:
        Tensor: With shape `(num_boxes, num_regressions)`,
            concatenated class and box outputs.
    """
    class_outputs = branch_tensors[0]
    boxes_outputs = branch_tensors[1]
    classification_layers, regression_layers = [], []
    for level in range(0, num_levels):
        class_leaf = class_outputs[level]
        class_leaf = Flatten()(class_leaf)
        classification_layers.append(class_leaf)

        regress_leaf = boxes_outputs[level]
        regress_leaf = Flatten()(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = Concatenate(axis=1)(classification_layers)
    regressions = Concatenate(axis=1)(regression_layers)
    num_boxes = K.int_shape(regressions)[-1] // num_regressions
    classifications = Reshape((num_boxes, num_classes))(classifications)
    classifications = Activation('softmax')(classifications)
    regressions = Reshape((num_boxes, num_regressions))(regressions)
    outputs = Concatenate(axis=2, name='boxes')([regressions, classifications])
    return outputs
