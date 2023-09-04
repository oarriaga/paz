from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from paz.backend.anchors import build_anchors
from paz.models.detection.efficientdet.efficientnet import EFFICIENTNET
from anchors import build_translation_anchors
from efficientdet_blocks import build_detector_head, EfficientNet_to_BiFPN
from efficientdet_blocks_with_bug import BiFPN
from efficientpose_blocks import build_pose_estimator_head

WEIGHT_PATH = (
    '/home/manummk95/Desktop/paz/paz/examples/efficientpose/weights/')


def EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                  FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                  anchor_scale, fusion, return_base, model_name, EfficientNet,
                  subnet_iterations=1, subnet_repeats=3, num_scales=3,
                  aspect_ratios=[1.0, 2.0, 0.5], survival_rate=None,
                  num_dims=4, momentum=0.997, epsilon=0.0001,
                  activation='sigmoid', num_anchors=9, num_filters=64,
                  num_pose_dims=3):

    if base_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', base_weights)
    if head_weights not in ['LINEMOD_OCCLUDED', None]:
        raise ValueError('Invalid head_weights: ', head_weights)
    if (base_weights is None) and (head_weights == 'COCO'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

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

    if ((base_weights == 'COCO') and (head_weights == 'LINEMOD_OCCLUDED')):
        model_filename = '-'.join([model_name, str(base_weights),
                                   str(head_weights) + '_weights.hdf5'])

    if not ((base_weights is None) and (head_weights is None)):
        weights_path = WEIGHT_PATH + model_filename
        model.load_weights(weights_path)

    image_shape = image.shape[1:3].as_list()
    model.prior_boxes = build_anchors(
        image_shape, branches, num_scales, aspect_ratios, anchor_scale)

    model.translation_priors = build_translation_anchors(
        image_shape, branches, num_scales, aspect_ratios)
    return model


def EFFICIENTPOSEA(num_classes=8, base_weights='COCO',
                   head_weights='LINEMOD_OCCLUDED', input_shape=(512, 512, 3),
                   FPN_num_filters=64, FPN_cell_repeats=3, subnet_repeats=3,
                   subnet_iterations=1, box_class_repeats=3, anchor_scale=4.0,
                   fusion='fast', return_base=False,
                   model_name='efficientpose-a',
                   scaling_coefficients=(1.0, 1.0, 0.8)):

    image = Input(shape=input_shape, name='image')
    EfficientNetb0 = EFFICIENTNET(image, scaling_coefficients)
    model = EFFICIENTPOSE(image, num_classes, base_weights, head_weights,
                          FPN_num_filters, FPN_cell_repeats, box_class_repeats,
                          anchor_scale, fusion, return_base, model_name,
                          EfficientNetb0, subnet_iterations, subnet_repeats)
    return model
