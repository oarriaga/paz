from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from paz.backend.anchors import build_anchors
from paz.models.detection.efficientdet.efficientnet import EFFICIENTNET
from anchors import build_translation_anchors
from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_detector_head, EfficientNet_to_BiFPN, BiFPN)
from efficientpose_blocks import build_pose_estimator_head

WEIGHT_PATH = 'weights/'


def EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                  FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                  anchor_scale, fusion, return_base, model_name, EfficientNet,
                  subnet_iterations=1, subnet_repeats=3, num_scales=3,
                  aspect_ratios=[1.0, 2.0, 0.5], survival_rate=None,
                  num_dims=4, num_anchors=9, num_filters=64,
                  num_pose_dims=3):
    """Builds EfficientPose model.

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
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        survival_rate: Float, specifying survival probability.
        num_dims: Int, number of output dimensions to regress.
        num_anchors: List, number of combinations of anchor box's scale
            and aspect ratios.
        num_filters: Int, number of subnet filters.
        num_pose_dims: Int, number of pose dimensions.

    # Returns
        model: EfficientPose model.

    # References
        [EfficientPose: An efficient, accurate and scalable end-to-end
        6D multi object pose estimation approach](
            https://arxiv.org/pdf/2011.04307.pdf)
    """
    if base_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', base_weights)
    if head_weights not in ['LINEMOD_OCCLUDED', None]:
        raise ValueError('Invalid head_weights: ', head_weights)
    if (base_weights is None) and (head_weights == 'COCO'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    branches, middles, skips = EfficientNet_to_BiFPN(
        EfficientNet, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)

    if return_base:
        outputs = middles
    else:
        detection_outputs = build_detector_head(
            middles, num_classes, num_dims, aspect_ratios, num_scales,
            FPN_num_filters, box_class_repeats, survival_rate)

        pose_outputs = build_pose_estimator_head(
            middles, subnet_iterations, subnet_repeats,
            num_anchors, num_filters, num_pose_dims)

        outputs = [detection_outputs, pose_outputs]

    model = Model(inputs=image, outputs=outputs, name=model_name)

    if ((base_weights == 'COCO') and (head_weights == 'LINEMOD_OCCLUDED')):
        model_filename = '-'.join([model_name, str(base_weights),
                                   str(head_weights) + '_weights.hdf5'])

    elif ((base_weights == 'COCO') and (head_weights is None)):
        model_filename = '-'.join([model_name, str(base_weights),
                                   str(head_weights) + '_weights.hdf5'])

    if not ((base_weights is None) and (head_weights is None)):
        weights_path = WEIGHT_PATH + model_filename
        finetunning_model_names = ['efficientpose-a-COCO-None_weights.hdf5',
                                   'efficientpose-b-COCO-None_weights.hdf5',
                                   'efficientpose-c-COCO-None_weights.hdf5',
                                   'efficientpose-d-COCO-None_weights.hdf5',
                                   'efficientpose-e-COCO-None_weights.hdf5',
                                   'efficientpose-f-COCO-None_weights.hdf5',
                                   'efficientpose-g-COCO-None_weights.hdf5',
                                   'efficientpose-h-COCO-None_weights.hdf5']
        by_name = True if model_filename in finetunning_model_names else False
        print('Loading %s model weights' % weights_path)
        model.load_weights(weights_path, by_name=by_name)

    image_shape = image.shape[1:3].as_list()
    model.prior_boxes = build_anchors(
        image_shape, branches, num_scales, aspect_ratios, anchor_scale)

    model.translation_priors = build_translation_anchors(
        image_shape, branches, num_scales, aspect_ratios)
    return model


def EFFICIENTPOSEA(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED', input_shape=(512, 512, 3),
                   FPN_num_filters=64, FPN_cell_repeats=3, subnet_repeats=2,
                   subnet_iterations=1, box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientpose-a',
                   scaling_coefficients=(1.0, 1.0, 0.8)):
    """Instantiates EfficientPose-A model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-A model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb0 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb0, subnet_iterations, subnet_repeats)
    return model


def EFFICIENTPOSEB(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED', input_shape=(640, 640, 3),
                   FPN_num_filters=88, FPN_cell_repeats=4, subnet_repeats=2,
                   subnet_iterations=1, box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientpose-b',
                   scaling_coefficients=(1.0, 1.0, 0.8)):
    """Instantiates EfficientPose-B model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-B model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb1 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb1, subnet_iterations, subnet_repeats)
    return model


def EFFICIENTPOSEC(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED', input_shape=(768, 768, 3),
                   FPN_num_filters=112, FPN_cell_repeats=5, subnet_repeats=2,
                   subnet_iterations=1, box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientpose-c',
                   scaling_coefficients=(1.1, 1.2, 0.7)):
    """Instantiates EfficientPose-C model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-C model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb2 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb2, subnet_iterations, subnet_repeats)
    return model


def EFFICIENTPOSED(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED', input_shape=(896, 896, 3),
                   FPN_num_filters=160, FPN_cell_repeats=6, subnet_repeats=3,
                   subnet_iterations=2, box_class_repeats=4, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientpose-d',
                   scaling_coefficients=(1.2, 1.4, 0.7)):
    """Instantiates EfficientPose-D model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-D model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb3 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb3, subnet_iterations, subnet_repeats)
    return model


def EFFICIENTPOSEE(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED',
                   input_shape=(1024, 1024, 3), FPN_num_filters=224,
                   FPN_cell_repeats=7, subnet_repeats=3, subnet_iterations=2,
                   box_class_repeats=4, anchor_scale=4.0, fusion='fast',
                   return_base=False, model_name='efficientpose-e',
                   scaling_coefficients=(1.2, 1.4, 0.7)):
    """Instantiates EfficientPose-E model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-E model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb4 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb4, subnet_iterations, subnet_repeats)
    return model


def EFFICIENTPOSEF(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED',
                   input_shape=(1280, 1280, 3), FPN_num_filters=288,
                   FPN_cell_repeats=7, subnet_repeats=3, subnet_iterations=2,
                   box_class_repeats=4, anchor_scale=4.0, fusion='fast',
                   return_base=False, model_name='efficientpose-f',
                   scaling_coefficients=(1.6, 2.2, 0.6)):
    """Instantiates EfficientPose-F model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-F model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb5 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb5, subnet_iterations, subnet_repeats)
    return model


def EFFICIENTPOSEG(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED',
                   input_shape=(1280, 1280, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, subnet_repeats=4, subnet_iterations=3,
                   box_class_repeats=5, anchor_scale=5.0, fusion='sum',
                   return_base=False, model_name='efficientpose-g',
                   scaling_coefficients=(1.8, 2.6, 0.5)):
    """Instantiates EfficientPose-G model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-G model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb6 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb6, subnet_iterations, subnet_repeats)
    return model


def EFFICIENTPOSEH(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED',
                   input_shape=(1536, 1536, 3), FPN_num_filters=384,
                   FPN_cell_repeats=8, subnet_repeats=4, subnet_iterations=3,
                   box_class_repeats=5, anchor_scale=5.0, fusion='sum',
                   return_base=False, model_name='efficientpose-h',
                   scaling_coefficients=(1.8, 2.6, 0.5)):
    """Instantiates EfficientPose-H model.

    # Arguments
        num_classes: Int, number of object classes.
        base_weights: Str, base weights name.
        head_weights: Str, head weights name.
        input_shape: Tuple, holding input image size.
        FPN_num_filters: Int, number of FPN filters.
        FPN_cell_repeats: Int, number of FPN blocks.
        subnet_repeats: Int, number of layers used in subnetworks.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        anchor_scale: Int, number of anchor scales.
        fusion: Str, feature fusion weighting method.
        return_base: Bool, whether to return base or not.
        model_name: Str, EfficientDet model name.
        scaling_coefficients: Tuple, EfficientNet scaling coefficients.

    # Returns
        model: EfficientPose-H model.
    """
    image = Input(shape=input_shape, name='image')
    EfficientNetb6 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb6, subnet_iterations, subnet_repeats)
    return model
