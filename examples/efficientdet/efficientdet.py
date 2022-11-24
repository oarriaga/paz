from efficientdet_model import EfficientDet


def EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                 FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                 anchor_scale, min_level, max_level, fusion,
                 return_base, model_name, backbone):
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
    model = EfficientDet(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD0(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(512, 512, 3), FPN_num_filters=64,
                   FPN_cell_repeats=3, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d0',
                   backbone='efficientnet-b0'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD1(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(640, 640, 3), FPN_num_filters=88,
                   FPN_cell_repeats=4, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d1',
                   backbone='efficientnet-b1'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD2(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(768, 768, 3), FPN_num_filters=112,
                   FPN_cell_repeats=5, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d2',
                   backbone='efficientnet-b2'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD3(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(896, 896, 3), FPN_num_filters=160,
                   FPN_cell_repeats=6, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d3',
                   backbone='efficientnet-b3'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD4(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1024, 1024, 3), FPN_num_filters=224,
                   FPN_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d4',
                   backbone='efficientnet-b4'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD5(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1280, 1280, 3), FPN_num_filters=288,
                   FPN_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fusion='fast',
                   return_base=False, model_name='efficientdet-d5',
                   backbone='efficientnet-b5'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD6(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1280, 1280, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   min_level=3, max_level=7, fusion='sum',
                   return_base=False, model_name='efficientdet-d6',
                   backbone='efficientnet-b6'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD7(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(1536, 1536, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   min_level=3, max_level=7, fusion='sum',
                   return_base=False, model_name='efficientdet-d7',
                   backbone='efficientnet-b6'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model


def EFFICIENTDETD7x(num_classes=90, base_weights='COCO', head_weights='COCO',
                    input_shape=(1536, 1536, 3), FPN_num_filters=384,
                    FPN_cell_repeats=8, box_class_repeats=5, anchor_scale=4.0,
                    min_level=3, max_level=8, fusion='sum',
                    return_base=False, model_name='efficientdet-d7x',
                    backbone='efficientnet-b7'):
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
    model = EFFICIENTDET(num_classes, base_weights, head_weights, input_shape,
                         FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                         anchor_scale, min_level, max_level, fusion,
                         return_base, model_name, backbone)
    return model
