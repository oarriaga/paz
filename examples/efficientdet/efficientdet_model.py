from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from anchors import get_prior_boxes
from efficientdet_blocks import ResampleFeatureMap
from efficientdet_blocks import FPNCells, ClassNet, BoxNet
from utils import create_multibox_head
from efficientnet_model import EfficientNet

import h5py
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

WEIGHT_PATH = (
    '/media/deepan/externaldrive1/project_repos/paz_versions'
    '/paz_efficientdet_weights/')

def EfficientDet(num_classes, base_weights, head_weights, input_shape,
                 fpn_num_filters, fpn_cell_repeats, box_class_repeats,
                 anchor_scale, min_level, max_level, fpn_weight_method,
                 return_base, model_name, backbone, training=False,
                 num_scales=3, aspect_ratios=[1.0, 2.0, 0.5],
                 survival_rate=None):
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
        name: Module name
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
    if (base_weights == 'COCO') and (head_weights is None):
        raise NotImplementedError('Invalid `base_weights` with head_weights')
    if (base_weights is None) and (head_weights == 'COCO'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    image = Input(shape=input_shape, name='image')

    branch_tensors = EfficientNet(image, backbone, input_shape)
    feature_levels = branch_tensors[min_level - 1: max_level + 1]

    for level in range(6, max_level + 1):
        resampler = ResampleFeatureMap(
            (level - min_level), fpn_num_filters, name='resample_p%d' % level)(
            feature_levels[-1], training, None)
        feature_levels.append(resampler)

    fpn_features = FPNCells(min_level, max_level, fpn_weight_method,
                            fpn_cell_repeats, fpn_num_filters
                            )(feature_levels, training)

    num_anchors = len(aspect_ratios) * num_scales
    class_outputs = ClassNet(fpn_features, num_classes, num_anchors,
                             fpn_num_filters, min_level, max_level,
                             box_class_repeats, survival_rate, training)
    box_outputs = BoxNet(fpn_features, num_anchors, fpn_num_filters,
                         min_level, max_level, box_class_repeats,
                         survival_rate, training)

    branch_tensors = [class_outputs, box_outputs]
    if return_base:
        outputs = branch_tensors
    else:
        num_levels = max_level - min_level + 1
        outputs = create_multibox_head(branch_tensors, num_levels, num_classes)
    model = Model(inputs=image, outputs=outputs, name=model_name)

    if (base_weights == 'COCO') and (head_weights == 'COCO'):
        weights_path = WEIGHT_PATH + model_name + '.h5'
        pretrained_weights = read_hdf5(weights_path)
        # for i in pretrained_weights:
        #     print(i, pretrained_weights[i].shape)
        # for n, i in enumerate(model.weights):
        #     print(i.name, "", model.weights[n].shape)
        layers = ['efficientnet-b0', 'fpn_cells', 'class_net', 'box_net',
                  'resample_p6']
        for n, i in enumerate(model.weights):
            name_str = i.name.split('/')[0]
            if name_str == 'efficientnet-b0':
                appender = i.name.split('/')[:-1]
                appender_str = '/'.join(appender)
                new_name = '/' + appender_str + '/' + i.name
            else:
                new_name = '/' + name_str + '/' + i.name
            if new_name in pretrained_weights.keys():
                # print('ADDING: ', new_name)
                if model.weights[n].shape == pretrained_weights[new_name].shape:
                    model.weights[n].assign(pretrained_weights[new_name])
            else:
                print('NOT ADDING: ', new_name)
                raise ValueError('NOT ADDING')
        # model.load_weights(weights_path)
    model.prior_boxes = get_prior_boxes(
        min_level, max_level, num_scales, aspect_ratios, anchor_scale,
        input_shape[0])
    return model
