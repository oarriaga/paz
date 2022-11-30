from tensorflow.keras.layers import Input

from efficientdet_model import EfficientDet
from efficientnet import EFFICIENTNET


def EFFICIENTDET(image, num_classes, base_weights, head_weights, input_shape,
                 FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                 anchor_scale, min_level, max_level, fusion,
                 return_base, model_name, EfficientNet):
    """ Creates EfficientDet model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet model.
    """
    model = EfficientDet(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNet)
    return model


def EFFICIENTDETD0(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(512, 512, 3), FPN_num_filters=64,
                   FPN_cell_repeats=3, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d0',
                   scaling_coefficients=(1.0, 1.0, 0.8)):
    """ Instantiates EfficientDet-D0 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D0 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb0 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb0)
    return model


def EFFICIENTDETD1(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(640, 640, 3), FPN_num_filters=88,
                   FPN_cell_repeats=4, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d1',
                   scaling_coefficients=(1.0, 1.1, 0.8)):
    """ Instantiates EfficientDet-D1 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D1 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb1 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb1)
    return model


def EFFICIENTDETD2(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(768, 768, 3), FPN_num_filters=112,
                   FPN_cell_repeats=5, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d2',
                   scaling_coefficients=(1.1, 1.2, 0.7)):
    """ Instantiate EfficientDet-D2 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D2 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb2 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb2)
    return model


def EFFICIENTDETD3(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(896, 896, 3), FPN_num_filters=160,
                   FPN_cell_repeats=6, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d3',
                   scaling_coefficients=(1.2, 1.4, 0.7)):
    """ Instantiates EfficientDet-D3 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D3 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb3 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb3)
    return model


def EFFICIENTDETD4(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1024, 1024, 3), FPN_num_filters=224,
                   FPN_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d4',
                   scaling_coefficients=(1.4, 1.8, 0.6)):
    """ Instantiates EfficientDet-D4 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D4 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb4 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb4)
    return model


def EFFICIENTDETD5(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1280, 1280, 3), FPN_num_filters=288,
                   FPN_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d5',
                   scaling_coefficients=(1.6, 2.2, 0.6)):
    """ Instantiates EfficientDet-D5 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D5 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb5 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb5)
    return model


def EFFICIENTDETD6(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1280, 1280, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   min_level=3, max_level=7, fusion='sum',
                   return_base=False, model_name='efficientdet-d6',
                   scaling_coefficients=(1.8, 2.6, 0.5)):
    """ Instantiates EfficientDet-D6 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D6 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb6 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb6)
    return model


def EFFICIENTDETD7(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1536, 1536, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   min_level=3, max_level=7, fusion='sum',
                   return_base=False, model_name='efficientdet-d7',
                   scaling_coefficients=(1.8, 2.6, 0.5)):
    """ Instantiates EfficientDet-D7 model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D7 model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb6 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb6)
    return model


def EFFICIENTDETD7x(num_classes=90, base_weights='COCO', head_weights='COCO',
                    input_shape=(1536, 1536, 3), FPN_num_filters=384,
                    FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=4.0,
                    min_level=3, max_level=8, fusion='sum',
                    return_base=False, model_name='efficientdet-d7x',
                    scaling_coefficients=(2.0, 3.1, 0.5)):
    """ Instantiates EfficientDet-D7x model.

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
        min_level: Int, minimum feature level.
        max_level: Int, maximum feature level.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        backbone: Str, EfficientNet backbone name.

    # Returns
        model: EfficientDet-D7x model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb7 = EFFICIENTNET(image, scaling_coefficients, input_shape)
    model = EFFICIENTDET(image, num_classes, base_weights, head_weights,
                         input_shape, FPN_num_filters, FPN_cell_repeats,
                         box_class_repeats, anchor_scale, min_level, max_level,
                         fusion, return_base, model_name, EfficientNetb7)
    return model
