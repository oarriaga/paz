import pytest
from tensorflow.keras.layers import Input
from paz.models.detection.efficientdet.efficientnet import EFFICIENTNET
from paz.models.detection.efficientdet.efficientdet_blocks import (
    EfficientNet_to_BiFPN, BiFPN)
from paz.models.pose_estimation.efficientpose.efficientpose_blocks import (
    RotationNet, TranslationNet)


@pytest.mark.parametrize(('input_shape, scaling_coefficients, FPN_num_filters,'
                          'FPN_cell_repeats, fusion, subnet_iterations,'
                          'subnet_repeats, output_shapes'),
                         [
                            (512,  (1.0, 1.0, 0.8), 64, 3, 'fast', 1, 2,
                                (36864, 9216, 2304, 576, 144)),
                            (640,  (1.0, 1.1, 0.8), 88, 4, 'fast', 1, 2,
                                (57600, 14400, 3600, 900, 225)),
                            (768,  (1.1, 1.2, 0.7), 112, 5, 'fast', 1, 2,
                                (82944, 20736, 5184, 1296, 324)),
                            (896,  (1.2, 1.4, 0.7), 160, 6, 'fast', 2, 3,
                                (112896, 28224, 7056, 1764, 441)),
                            (1024, (1.4, 1.8, 0.6), 224, 7, 'fast', 2, 3,
                                (147456, 36864, 9216, 2304, 576)),
                            (1280, (1.6, 2.2, 0.6), 288, 7, 'fast', 2, 3,
                                (230400, 57600, 14400, 3600, 900)),
                            (1280, (1.8, 2.6, 0.5), 384, 8, 'sum', 3, 4,
                                (230400, 57600, 14400, 3600, 900)),
                            (1536, (1.8, 2.6, 0.5), 384, 8, 'sum', 3, 4,
                                (331776, 82944, 20736, 5184, 1296))
                         ])
def test_EfficientPose_RotationNet(input_shape, scaling_coefficients,
                                   FPN_num_filters, FPN_cell_repeats, fusion,
                                   subnet_iterations, subnet_repeats,
                                   output_shapes):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(
        branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    num_dims, num_anchors, num_filters = (3, 9, 64)
    args = (middles, subnet_iterations, subnet_repeats, num_anchors)
    rotations = RotationNet(*args, num_filters, num_dims)
    assert len(rotations) == 5, 'Rotation output length fail'
    for rotation, output_shape in zip(rotations, output_shapes):
        assert rotation.shape == (None, output_shape, 3), (
            'Rotation outputs shape fail')
    del branch_tensors, branches, middles, skips, rotations


@pytest.mark.parametrize(('input_shape, scaling_coefficients, FPN_num_filters,'
                          'FPN_cell_repeats, fusion, subnet_iterations,'
                          'subnet_repeats, output_shapes'),
                         [
                            (512,  (1.0, 1.0, 0.8), 64, 3, 'fast', 1, 2,
                                (36864, 9216, 2304, 576, 144)),
                            (640,  (1.0, 1.1, 0.8), 88, 4, 'fast', 1, 2,
                                (57600, 14400, 3600, 900, 225)),
                            (768,  (1.1, 1.2, 0.7), 112, 5, 'fast', 1, 2,
                                (82944, 20736, 5184, 1296, 324)),
                            (896,  (1.2, 1.4, 0.7), 160, 6, 'fast', 2, 3,
                                (112896, 28224, 7056, 1764, 441)),
                            (1024, (1.4, 1.8, 0.6), 224, 7, 'fast', 2, 3,
                                (147456, 36864, 9216, 2304, 576)),
                            (1280, (1.6, 2.2, 0.6), 288, 7, 'fast', 2, 3,
                                (230400, 57600, 14400, 3600, 900)),
                            (1280, (1.8, 2.6, 0.5), 384, 8, 'sum', 3, 4,
                                (230400, 57600, 14400, 3600, 900)),
                            (1536, (1.8, 2.6, 0.5), 384, 8, 'sum', 3, 4,
                                (331776, 82944, 20736, 5184, 1296))
                         ])
def test_EfficientPose_TranslationNet(input_shape, scaling_coefficients,
                                      FPN_num_filters, FPN_cell_repeats,
                                      fusion, subnet_iterations,
                                      subnet_repeats, output_shapes):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(
        branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    num_anchors, num_filters = (9, 64)
    args = (middles, subnet_iterations, subnet_repeats, num_anchors)
    translations = TranslationNet(*args, num_filters)
    assert len(translations) == 5, 'Translation output length fail'
    for translation, output_shape in zip(translations, output_shapes):
        assert translation.shape == (None, output_shape, 3), (
            'Translation outputs shape fail')
    del branch_tensors, branches, middles, skips, translations
