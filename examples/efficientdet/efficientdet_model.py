from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file

from anchors import build_prior_boxes
from efficientdet_blocks import BiFPN, BoxNet, ClassNet
from efficientnet_model import EfficientNet
from utils import create_multibox_head

WEIGHT_PATH = (
    'https://github.com/oarriaga/altamira-data/releases/download/v0.16/')


def EfficientDet(num_classes, base_weights, head_weights, input_shape,
                 fpn_num_filters, fpn_cell_repeats, box_class_repeats,
                 anchor_scale, min_level, max_level, fpn_weight_method,
                 return_base, model_name, backbone, num_scales=3,
                 aspect_ratios=[1.0, 2.0, 0.5], survival_rate=None):
    """EfficientDet model in PAZ.
    # References
        -[Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)

    # Arguments
        image_size: Int, size of the input image.
        num_classes: Int, specifying the number of class in the
        output.
        fpn_num_filters: Int, FPN filter output size.
        fpn_cell_repeats: Int, Number of consecutive FPN block.
        box_class_repeats: Int, Number of consective regression
        and classification blocks.
        anchor_scale: Int, specifying the number of anchor
        scales.
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn.
        return_base: Bool, indicating the usage of features only
        from EfficientDet
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        training: Bool, whether EfficientDet architecture is trained.
        layer.
        num_scales: Int, specifying the number of scales in the
        anchor boxes.
        aspect_ratios: List, specifying the aspect ratio of the
        survival_rate: Float, specifying the survival probability

    # Returns
        model: EfficientDet model specified in model_name with the following:
        class_outputs: Tensor, Logits for all classes corresponding to
        the features associated with the box coordinates.
        box_outputs: Tensor,  Box coordinate offsets for the
        corresponding prior boxes.
    """
    if base_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', base_weights)
    if head_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', head_weights)
    if (base_weights is None) and (head_weights == 'COCO'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    image = Input(shape=input_shape, name='image')
    branch_tensors = EfficientNet(image, backbone, input_shape)
    for fpn_cell_id in range(fpn_cell_repeats):
        branch_tensors = BiFPN(branch_tensors, fpn_num_filters,
                               fpn_cell_id, fpn_weight_method)
    num_anchors = len(aspect_ratios) * num_scales
    class_outputs = ClassNet(branch_tensors, num_classes, num_anchors,
                             fpn_num_filters, min_level, max_level,
                             box_class_repeats, survival_rate)
    box_outputs = BoxNet(branch_tensors, num_anchors, fpn_num_filters,
                         min_level, max_level, box_class_repeats,
                         survival_rate)

    branch_tensors = [class_outputs, box_outputs]
    if return_base:
        outputs = branch_tensors
    else:
        num_levels = max_level - min_level + 1
        outputs = create_multibox_head(branch_tensors, num_levels, num_classes)
    model = Model(inputs=image, outputs=outputs, name=model_name)

    if (((base_weights == 'COCO') and (head_weights == 'COCO')) or
            ((base_weights == 'COCO') and (head_weights is None))):
        model_filename = (model_name + '-' + str(base_weights) + '-' +
                          str(head_weights) + '_weights.hdf5')
        weights_path = get_file(model_filename, WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        model.load_weights(weights_path)

    model.prior_boxes = build_prior_boxes(
        min_level, max_level, num_scales, aspect_ratios, anchor_scale,
        input_shape[0:2])
    return model
