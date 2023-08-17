from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from paz.backend.anchors import build_anchors
from efficientdet_blocks import (
    BiFPN, build_detector_head, EfficientNet_to_BiFPN)
from paz.models.detection.efficientdet.efficientnet import EFFICIENTNET
from efficientpose_blocks import build_pose_estimator_head


def EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                  FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                  anchor_scale, fusion, return_base, model_name, EfficientNet,
                  subnet_iterations=1, subnet_repeats=3, num_scales=3,
                  aspect_ratios=[1.0, 2.0, 0.5], survival_rate=None,
                  num_dims=4, momentum=0.997, epsilon=0.0001,
                  activation='sigmoid', num_anchors=9, num_filters=64,
                  num_pose_dims=3):
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

    branches, middles, skips = EfficientNet_to_BiFPN(
        EfficientNet, FPN_num_filters, momentum, epsilon)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters,
                               fusion, momentum, epsilon)

    if return_base:
        outputs = middles
    else:
        detection_outputs = build_detector_head(
            middles, num_classes, num_dims, aspect_ratios, num_scales,
            FPN_num_filters, box_class_repeats, survival_rate,
            momentum, epsilon, activation)

        pose_outputs = build_pose_estimator_head(
            middles, subnet_iterations, subnet_repeats,
            num_anchors, num_filters, num_pose_dims)
        outputs = [detection_outputs, pose_outputs]

    model = Model(inputs=image, outputs=outputs, name=model_name)

    if not ((base_weights is None) and (head_weights is None)):
        weights_path = "weights/phi_0_occlusion_best_ADD(-S).h5"
        print('Loading %s model weights' % weights_path)
        model.load_weights(weights_path, by_name=True)

    image_shape = image.shape[1:3].as_list()
    model.prior_boxes = build_anchors(
        image_shape, branches, num_scales, aspect_ratios, anchor_scale)
    return model


def EFFICIENTPOSEA(num_classes=8, base_weights='COCO', head_weights='COCO',
                   input_shape=(512, 512, 3), FPN_num_filters=64,
                   FPN_cell_repeats=3, subnet_repeats=3, subnet_iterations=1,
                   box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientpose-a',
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
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb0, subnet_iterations, subnet_repeats)
    return model
