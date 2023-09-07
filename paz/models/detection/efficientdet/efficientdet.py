from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from ....backend.anchors import build_anchors
from .efficientdet_blocks import (
    BiFPN, build_detector_head, EfficientNet_to_BiFPN)
from .efficientnet import EFFICIENTNET

WEIGHT_PATH = (
    'https://github.com/oarriaga/altamira-data/releases/download/v0.16/')


def EFFICIENTDET(image, num_classes, base_weights, head_weights,
                 FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                 anchor_scale, fusion, return_base, model_name, EfficientNet,
                 num_scales=3, aspect_ratios=[1.0, 2.0, 0.5],
                 survival_rate=None, num_dims=4):
    """Creates EfficientDet model.

    # Arguments
        image: Tensor of shape `(batch_size, input_shape)`.
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        EfficientNet: List, containing branch tensors.
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        survival_rate: Float, specifying survival probability.
        num_dims: Int, number of output dimensions to regress.

    # Returns
        model: EfficientDet model.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    if base_weights not in ['COCO', 'VOC', None]:
        raise ValueError('Invalid base_weights: ', base_weights)
    if head_weights not in ['COCO', 'VOC', None]:
        raise ValueError('Invalid head_weights: ', head_weights)
    if (base_weights is None) and (head_weights == 'COCO'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')
    if (base_weights is None) and (head_weights == 'VOC'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    branches, middles, skips = EfficientNet_to_BiFPN(
        EfficientNet, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)

    if return_base:
        outputs = middles
    else:
        outputs = build_detector_head(
            middles, num_classes, num_dims, aspect_ratios, num_scales,
            FPN_num_filters, box_class_repeats, survival_rate)

    model = Model(inputs=image, outputs=outputs, name=model_name)

    if ((base_weights == 'COCO') and (head_weights == 'COCO')):
        model_filename = '-'.join([model_name, str(base_weights),
                                   str(head_weights) + '_weights.hdf5'])
    if ((base_weights == 'VOC') and (head_weights == 'VOC')):
        model_filename = '-'.join([model_name, str(base_weights),
                                   str(head_weights) + '_weights.hdf5'])
    elif ((base_weights == 'COCO') and (head_weights is None)):
        model_filename = '-'.join([model_name, str(base_weights),
                                   str(head_weights) + '_weights.hdf5'])

    if not ((base_weights is None) and (head_weights is None)):
        weights_path = get_file(model_filename, WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        finetunning_model_names = ['efficientdet-d0-COCO-None_weights.hdf5',
                                   'efficientdet-d1-COCO-None_weights.hdf5',
                                   'efficientdet-d2-COCO-None_weights.hdf5',
                                   'efficientdet-d3-COCO-None_weights.hdf5',
                                   'efficientdet-d4-COCO-None_weights.hdf5',
                                   'efficientdet-d5-COCO-None_weights.hdf5',
                                   'efficientdet-d6-COCO-None_weights.hdf5',
                                   'efficientdet-d7-COCO-None_weights.hdf5']
        by_name = True if model_filename in finetunning_model_names else False
        model.load_weights(weights_path, by_name=by_name)

    image_shape = image.shape[1:3].as_list()
    model.prior_boxes = build_anchors(
        image_shape, branches, num_scales, aspect_ratios, anchor_scale)
    return model


def EFFICIENTDETD0(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(512, 512, 3), FPN_num_filters=64,
                   FPN_cell_repeats=3, box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientdet-d0',
                   scaling_coefficients=(1.0, 1.0, 0.8)):
    """Instantiates EfficientDet-D0 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D0 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb0 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb0)
    return model


def EFFICIENTDETD1(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(640, 640, 3), FPN_num_filters=88,
                   FPN_cell_repeats=4, box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientdet-d1',
                   scaling_coefficients=(1.0, 1.1, 0.8)):
    """Instantiates EfficientDet-D1 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D1 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb1 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb1)
    return model


def EFFICIENTDETD2(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(768, 768, 3), FPN_num_filters=112,
                   FPN_cell_repeats=5, box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientdet-d2',
                   scaling_coefficients=(1.1, 1.2, 0.7)):
    """Instantiate EfficientDet-D2 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D2 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb2 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb2)
    return model


def EFFICIENTDETD3(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(896, 896, 3), FPN_num_filters=160,
                   FPN_cell_repeats=6, box_class_repeats=4, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientdet-d3',
                   scaling_coefficients=(1.2, 1.4, 0.7)):
    """Instantiates EfficientDet-D3 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D3 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb3 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb3)
    return model


def EFFICIENTDETD4(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1024, 1024, 3), FPN_num_filters=224,
                   FPN_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientdet-d4',
                   scaling_coefficients=(1.4, 1.8, 0.6)):
    """Instantiates EfficientDet-D4 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D4 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb4 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb4)
    return model


def EFFICIENTDETD5(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1280, 1280, 3), FPN_num_filters=288,
                   FPN_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientdet-d5',
                   scaling_coefficients=(1.6, 2.2, 0.6)):
    """Instantiates EfficientDet-D5 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D5 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb5 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb5)
    return model


def EFFICIENTDETD6(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1280, 1280, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   fusion='sum', return_base=False,
                   model_name='efficientdet-d6',
                   scaling_coefficients=(1.8, 2.6, 0.5)):
    """Instantiates EfficientDet-D6 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D6 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb6 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb6)
    return model


def EFFICIENTDETD7(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1536, 1536, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   fusion='sum', return_base=False,
                   model_name='efficientdet-d7',
                   scaling_coefficients=(1.8, 2.6, 0.5)):
    """Instantiates EfficientDet-D7 model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D7 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb6 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb6)
    return model


def EFFICIENTDETD7x(num_classes=90, base_weights='COCO', head_weights='COCO',
                    input_shape=(1536, 1536, 3), FPN_num_filters=384,
                    FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=4.0,
                    fusion='sum', return_base=False,
                    model_name='efficientdet-d7x',
                    scaling_coefficients=(2.0, 3.1, 0.5)):
    """Instantiates EfficientDet-D7x model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientDet-D7x model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb7 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, fusion, return_base, model_name,
                         EfficientNetb7)
    return model
