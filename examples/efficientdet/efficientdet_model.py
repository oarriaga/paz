from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file

from anchors import build_prior_boxes
from efficientdet_blocks import BiFPN, BoxNet, ClassNet, efficientnet_to_BiFPN
from efficientnet_model import efficientnet
from utils import create_multibox_head

WEIGHT_PATH = (
    'https://github.com/oarriaga/altamira-data/releases/download/v0.16/')


def EfficientDet(num_classes, base_weights, head_weights, input_shape,
                 FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                 anchor_scale, min_level, max_level, fusion,
                 return_base, model_name, backbone, num_scales=3,
                 aspect_ratios=[1.0, 2.0, 0.5], survival_rate=None):
    """EfficientDet model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, input image shape.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, anchor scale.
        min_level: Int, minimum features level.
        max_level: Int, maximum features level.
        fusion: Str, feature fusion method.
        return_base: Bool, use only EfficientDet features.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        survival_rate: Float, specifying survival probability.

    # Returns
        model: EfficientDet model.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    if base_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', base_weights)
    if head_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', head_weights)
    if (base_weights is None) and (head_weights == 'COCO'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    image = Input(shape=input_shape, name='image')
    branches = efficientnet(image, backbone, input_shape)

    middles, skips = efficientnet_to_BiFPN(branches, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)

    num_anchors = len(aspect_ratios) * num_scales
    args = (middles, num_anchors, FPN_num_filters, min_level,
            max_level, box_class_repeats, survival_rate)
    class_outputs = ClassNet(*args, num_classes)
    box_outputs = BoxNet(*args)

    branches = [class_outputs, box_outputs]
    if return_base:
        outputs = branches
    else:
        num_levels = max_level - min_level + 1
        outputs = create_multibox_head(branches, num_levels, num_classes)

    model = Model(inputs=image, outputs=outputs, name=model_name)

    if ((base_weights == 'COCO') and (head_weights == 'COCO')):
        model_filename = (model_name + '-' + str(base_weights) + '-' +
                          str(head_weights) + '_weights.hdf5')
    elif ((base_weights == 'COCO') and (head_weights is None)):
        model_filename = (model_name + '-' + str(base_weights) + '-' +
                          str(head_weights) + '_weights.hdf5')

    weights_path = get_file(model_filename, WEIGHT_PATH + model_filename,
                            cache_subdir='paz/models')
    print('Loading %s model weights' % weights_path)
    model.load_weights(weights_path)

    args = (min_level, max_level, num_scales, aspect_ratios,
            anchor_scale, input_shape[0:2])
    model.prior_boxes = build_prior_boxes(*args)
    return model
