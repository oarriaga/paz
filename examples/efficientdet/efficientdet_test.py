import pytest
import os
import numpy as np
import tensorflow as tf
from paz.backend.image import load_image
from keras.utils.layer_utils import count_params
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import get_file
from paz.models.detection.efficientdet import (
    EFFICIENTDETD0, EFFICIENTDETD1, EFFICIENTDETD2, EFFICIENTDETD3,
    EFFICIENTDETD4, EFFICIENTDETD5, EFFICIENTDETD6, EFFICIENTDETD7,
    EfficientNet_to_BiFPN, BiFPN)
from paz.models.detection.efficientdet.efficientnet import (
    EFFICIENTNET, apply_drop_connect, conv_block, MBconv_blocks)
from paz.models.detection.efficientdet.efficientdet_blocks import (
    ClassNet, BoxesNet)
from paz.models.detection.efficientdet.layers import FuseFeature
from paz.abstract.messages import Box2D
from paz.pipelines import (
    EFFICIENTDETD0COCO, EFFICIENTDETD1COCO, EFFICIENTDETD2COCO,
    EFFICIENTDETD3COCO, EFFICIENTDETD4COCO, EFFICIENTDETD5COCO,
    EFFICIENTDETD6COCO, EFFICIENTDETD7COCO)


@pytest.fixture
def model_input_name():
    return 'image'


@pytest.fixture
def model_output_name():
    return 'boxes'


@pytest.fixture
def model_weight_path():
    WEIGHT_PATH = (
        'https://github.com/oarriaga/altamira-data/releases/download/v0.16/')
    return WEIGHT_PATH


@pytest.fixture
def image_with_multiple_objects():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download/v0.16/'
           'image_with_multiple_objects.png')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


def boxes_EFFICIENTDETD0COCO():
    boxes2D = [
        Box2D(np.array([208, 88, 625, 473]), 0.91638654, 'person'),
        Box2D(np.array([135, 65, 388, 262]), 0.93165081, 'dog'),
        Box2D(np.array([0, 81, 157, 238]), 0.78440314, 'potted plant'),
        Box2D(np.array([27, 153, 197, 469]), 0.74715495, 'tv'),
        Box2D(np.array([178, 269, 304, 325]), 0.81094884, 'mouse'),
        Box2D(np.array([216, 301, 414, 473]), 0.81328964, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD1COCO():
    boxes2D = [
        Box2D(np.array([205, 86, 632, 476]), 0.96527081, 'person'),
        Box2D(np.array([132, 58, 390, 265]), 0.97670412, 'dog'),
        Box2D(np.array([2, 81, 151, 243]), 0.74967992, 'potted plant'),
        Box2D(np.array([33, 159, 199, 456]), 0.88773757, 'tv'),
        Box2D(np.array([184, 271, 239, 305]), 0.81249493, 'mouse'),
        Box2D(np.array([176, 269, 302, 331]), 0.74713963, 'mouse'),
        Box2D(np.array([221, 307, 412, 477]), 0.95043092, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD2COCO():
    boxes2D = [
        Box2D(np.array([226, 78, 627, 475]), 0.96915191, 'person'),
        Box2D(np.array([137, 61, 387, 263]), 0.89453774, 'dog'),
        Box2D(np.array([418, 113, 469, 215]), 0.65446805, 'chair'),
        Box2D(np.array([2, 86, 150, 228]), 0.70579999, 'potted plant'),
        Box2D(np.array([189, 273, 240, 306]), 0.83571755, 'mouse'),
        Box2D(np.array([254, 280, 303, 312]), 0.73552852, 'mouse'),
        Box2D(np.array([180, 269, 301, 324]), 0.62722778, 'mouse'),
        Box2D(np.array([221, 309, 411, 478]), 0.96427386, 'keyboard'),
        Box2D(np.array([13, 397, 169, 478]), 0.62784308, 'book')
        ]
    return boxes2D


def boxes_EFFICIENTDETD3COCO():
    boxes2D = [
        Box2D(np.array([200, 77, 628, 474]), 0.95490562, 'person'),
        Box2D(np.array([136, 61, 391, 261]), 0.97604763, 'dog'),
        Box2D(np.array([417, 112, 469, 216]), 0.77754944, 'chair'),
        Box2D(np.array([0, 84, 153, 220]), 0.88959991, 'potted plant'),
        Box2D(np.array([27, 150, 201, 466]), 0.84968209, 'tv'),
        Box2D(np.array([187, 274, 241, 306]), 0.91144222, 'mouse'),
        Box2D(np.array([258, 281, 304, 313]), 0.80733084, 'mouse'),
        Box2D(np.array([223, 307, 413, 477]), 0.95095759, 'keyboard'),
        Box2D(np.array([9, 396, 169, 476]), 0.68497151, 'book'),
        Box2D(np.array([460, 412, 483, 452]), 0.69323199, 'clock')
        ]
    return boxes2D


def boxes_EFFICIENTDETD4COCO():
    boxes2D = [
        Box2D(np.array([196, 80, 628, 476]), 0.99412435, 'person'),
        Box2D(np.array([136, 61, 389, 261]), 0.99221706, 'dog'),
        Box2D(np.array([417, 112, 468, 216]), 0.79600876, 'chair'),
        Box2D(np.array([0, 83, 152, 221]), 0.93628972, 'potted plant'),
        Box2D(np.array([29, 148, 198, 463]), 0.88414156, 'tv'),
        Box2D(np.array([185, 274, 243, 307]), 0.77017039, 'mouse'),
        Box2D(np.array([235, 279, 303, 310]), 0.72798311, 'mouse'),
        Box2D(np.array([220, 309, 409, 477]), 0.97034329, 'keyboard'),
        Box2D(np.array([10, 395, 170, 477]), 0.84240531, 'book'),
        Box2D(np.array([459, 411, 483, 451]), 0.70187550, 'clock')
        ]
    return boxes2D


def boxes_EFFICIENTDETD5COCO():
    boxes2D = [
        Box2D(np.array([184, 76, 632, 475]), 0.96830463, 'person'),
        Box2D(np.array([137, 61, 388, 259]), 0.98346567, 'dog'),
        Box2D(np.array([416, 112, 468, 217]), 0.73830294, 'chair'),
        Box2D(np.array([0, 83, 153, 221]), 0.87367552, 'potted plant'),
        Box2D(np.array([30, 158, 194, 459]), 0.89623433, 'tv'),
        Box2D(np.array([188, 274, 241, 306]), 0.88030225, 'mouse'),
        Box2D(np.array([247, 280, 304, 313]), 0.72541850, 'mouse'),
        Box2D(np.array([218, 310, 410, 477]), 0.98626524, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD6COCO():
    boxes2D = [
        Box2D(np.array([132, 32, 688, 520]), 0.97478908, 'person'),
        Box2D(np.array([102, 34, 420, 281]), 0.98087191, 'dog'),
        Box2D(np.array([-20, 67, 171, 241]), 0.85821670, 'potted plant'),
        Box2D(np.array([11, 115, 222, 509]), 0.86900913, 'tv'),
        Box2D(np.array([183, 269, 249, 311]), 0.87571144, 'mouse'),
        Box2D(np.array([199, 288, 434, 496]), 0.98552298, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD7COCO():
    boxes2D = [
        Box2D(np.array([196, 77, 630, 474]), 0.98556220, 'person'),
        Box2D(np.array([137, 63, 391, 260]), 0.99482345, 'dog'),
        Box2D(np.array([342, 111, 470, 366]), 0.84057724, 'chair'),
        Box2D(np.array([0, 84, 153, 221]), 0.96013510, 'potted plant'),
        Box2D(np.array([32, 155, 197, 466]), 0.92974454, 'tv'),
        Box2D(np.array([191, 274, 240, 306]), 0.94579493, 'mouse'),
        Box2D(np.array([245, 280, 304, 315]), 0.69237989, 'mouse'),
        Box2D(np.array([220, 310, 409, 477]), 0.99352794, 'keyboard')
        ]
    return boxes2D


def get_test_images(image_size, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return tf.zeros((batch_size, image_size, image_size, 3), dtype=tf.float32)


def get_EfficientNet_hyperparameters():
    efficientnet_hyperparameters = {
        "intro_filters": [32, 16, 24, 40, 80, 112, 192],
        "outro_filters": [16, 24, 40, 80, 112, 192, 320],
        "D_divisor": 8,
        "kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
        "repeats": [1, 2, 2, 3, 3, 4, 1],
        "excite_ratio": 0.25,
        "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [1, 1], [2, 2], [1, 1]],
        "expand_ratios": [1, 6, 6, 6, 6, 6, 6]
    }
    return efficientnet_hyperparameters


@pytest.mark.parametrize(('input_shape, dtype, target_shape, is_training'),
                         [
                            ((1, 1), tf.dtypes.float64, ([1, 1, 1, 1]), True),
                            ((5, 5), tf.dtypes.float64, ([5, 1, 5, 5]), True),
                            ((3, 5), tf.dtypes.float64, ([3, 1, 3, 5]), True),
                            ((5, 3), tf.dtypes.float64, ([5, 1, 5, 3]), True),
                            ((1, 1), tf.dtypes.float64, (1, 1), False),
                            ((5, 5), tf.dtypes.float64, (5, 5), False),
                            ((3, 5), tf.dtypes.float64, (3, 5), False),
                            ((5, 3), tf.dtypes.float64, (5, 3), False)
                         ])
def test_drop_connect(input_shape, dtype, target_shape, is_training):
    x = tf.random.uniform(input_shape, minval=0, maxval=5, dtype=dtype)
    survival_rate = np.random.uniform(0.0, 1.0)
    y = apply_drop_connect(x, is_training, survival_rate)
    assert y.shape == target_shape, 'Incorrect target shape'
    assert y.dtype == dtype, 'Incorrect target datatype'
    del x, y


@pytest.mark.parametrize(('image_size, scaling_coefficients, output_shape'),
                         [
                            (512,  (1.0, 1.0, 0.8), (256, 256, 16)),
                            (640,  (1.0, 1.1, 0.8), (320, 320, 16)),
                            (768,  (1.1, 1.2, 0.7), (384, 384, 16)),
                            (896,  (1.2, 1.4, 0.7), (448, 448, 24)),
                            (1024, (1.4, 1.8, 0.6), (512, 512, 24)),
                            (1280, (1.6, 2.2, 0.6), (640, 640, 24)),
                            (1280, (1.8, 2.6, 0.5), (640, 640, 32)),
                            (1536, (1.8, 2.6, 0.5), (768, 768, 32)),
                         ])
def test_EfficientNet_bottleneck_block(image_size, scaling_coefficients,
                                       output_shape):
    shape = (image_size, image_size, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    assert branch_tensors[0].shape == (None, ) + output_shape, (
        'Bottleneck block output shape mismatch')
    del image, branch_tensors


@pytest.mark.parametrize(('image_size, scaling_coefficients, output_shape'),
                         [
                            (512,  (1.0, 1.0, 0.8), (256, 256, 16)),
                            (640,  (1.0, 1.1, 0.8), (320, 320, 16)),
                            (768,  (1.1, 1.2, 0.7), (384, 384, 16)),
                            (896,  (1.2, 1.4, 0.7), (448, 448, 24)),
                            (1024, (1.4, 1.8, 0.6), (512, 512, 24)),
                            (1280, (1.6, 2.2, 0.6), (640, 640, 24)),
                            (1280, (1.8, 2.6, 0.5), (640, 640, 32)),
                            (1536, (1.8, 2.6, 0.5), (768, 768, 32)),
                         ])
def test_EfficientNet_SE_block(image_size, scaling_coefficients,
                               output_shape):
    shape = (image_size, image_size, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients,
                                  excite_ratio=0.8)
    assert branch_tensors[0].shape == (None, ) + output_shape, (
        'SE block output shape mismatch')
    del image, branch_tensors


@pytest.mark.parametrize(('image_size, scaling_coefficients, output_shape'),
                         [
                            (512,  (1.0, 1.0, 0.8), (1, 256, 256, 32)),
                            (640,  (1.0, 1.1, 0.8), (1, 320, 320, 32)),
                            (768,  (1.1, 1.2, 0.7), (1, 384, 384, 32)),
                            (896,  (1.2, 1.4, 0.7), (1, 448, 448, 40)),
                            (1024, (1.4, 1.8, 0.6), (1, 512, 512, 48)),
                            (1280, (1.6, 2.2, 0.6), (1, 640, 640, 48)),
                            (1280, (1.8, 2.6, 0.5), (1, 640, 640, 56)),
                            (1536, (1.8, 2.6, 0.5), (1, 768, 768, 56)),
                         ])
def test_EfficientNet_conv_block(image_size, scaling_coefficients,
                                 output_shape):
    images = get_test_images(image_size, 1)
    efficientnet_hyperparameters = get_EfficientNet_hyperparameters()
    intro_filters = efficientnet_hyperparameters["intro_filters"]
    D_divisor = efficientnet_hyperparameters["D_divisor"]
    W_coefficient, D_coefficient, survival_rate = scaling_coefficients
    x = conv_block(images, intro_filters, W_coefficient, D_divisor)
    assert x.shape == output_shape, "Output shape mismatch"
    del images, x


@pytest.mark.parametrize(('image_size, scaling_coefficients, output_shape'),
                         [
                            (512, (1.0, 1.0, 0.8), [(1, 256, 256, 16),
                                                    (1, 128, 128, 24),
                                                    (1, 64, 64, 40),
                                                    (1, 32, 32, 112),
                                                    (1, 16, 16, 320)]),
                            (640, (1.0, 1.1, 0.8), [(1, 320, 320, 16),
                                                    (1, 160, 160, 24),
                                                    (1, 80, 80, 40),
                                                    (1, 40, 40, 112),
                                                    (1, 20, 20, 320)]),
                            (768, (1.1, 1.2, 0.7), [(1, 384, 384, 16),
                                                    (1, 192, 192, 24),
                                                    (1, 96, 96, 48),
                                                    (1, 48, 48, 120),
                                                    (1, 24, 24, 352)]),
                            (896,  (1.2, 1.4, 0.7), [(1, 448, 448, 24),
                                                     (1, 224, 224, 32),
                                                     (1, 112, 112, 48),
                                                     (1, 56, 56, 136),
                                                     (1, 28, 28, 384)]),
                            (1024, (1.4, 1.8, 0.6), [(1, 512, 512, 24),
                                                     (1, 256, 256, 32),
                                                     (1, 128, 128, 56),
                                                     (1, 64, 64, 160),
                                                     (1, 32, 32, 448)]),
                            (1280, (1.6, 2.2, 0.6), [(1, 640, 640, 24),
                                                     (1, 320, 320, 40),
                                                     (1, 160, 160, 64),
                                                     (1, 80, 80, 176),
                                                     (1, 40, 40, 512)]),
                            (1280, (1.8, 2.6, 0.5), [(1, 640, 640, 32),
                                                     (1, 320, 320, 40),
                                                     (1, 160, 160, 72),
                                                     (1, 80, 80, 200),
                                                     (1, 40, 40, 576)]),
                            (1536, (1.8, 2.6, 0.5), [(1, 768, 768, 32),
                                                     (1, 384, 384, 40),
                                                     (1, 192, 192, 72),
                                                     (1, 96, 96, 200),
                                                     (1, 48, 48, 576)])
                         ])
def test_EfficientNet_MBconv_blocks(image_size, scaling_coefficients,
                                    output_shape):
    images = get_test_images(image_size, 1)
    efficientnet_hyperparameters = get_EfficientNet_hyperparameters()
    intro_filters = efficientnet_hyperparameters["intro_filters"]
    D_divisor = efficientnet_hyperparameters["D_divisor"]
    kernel_sizes = efficientnet_hyperparameters["kernel_sizes"]
    outro_filters = efficientnet_hyperparameters["outro_filters"]
    repeats = efficientnet_hyperparameters["repeats"]
    excite_ratio = efficientnet_hyperparameters["excite_ratio"]
    strides = efficientnet_hyperparameters["strides"]
    expand_ratios = efficientnet_hyperparameters["expand_ratios"]
    W_coefficient, D_coefficient, survival_rate = scaling_coefficients
    x = conv_block(images, intro_filters, W_coefficient, D_divisor)
    x = MBconv_blocks(
        x, kernel_sizes, intro_filters, outro_filters,
        W_coefficient, D_coefficient, D_divisor, repeats,
        excite_ratio, survival_rate, strides, expand_ratios)
    assert len(x) == len(output_shape), "Feature count mismatch"
    for feature, target_shape in zip(x, output_shape):
        assert feature.shape == target_shape, "Feature shape mismatch"
    del images, x


@pytest.mark.parametrize(('input_shape, scaling_coefficients, feature_shape,'
                          'feature_channels'),
                         [
                            (512,  (1.0, 1.0, 0.8), (256, 128, 64, 32, 16),
                             (16, 24, 40, 112, 320)),
                            (640,  (1.0, 1.1, 0.8), (320, 160, 80, 40, 20),
                             (16, 24, 40, 112, 320)),
                            (768,  (1.1, 1.2, 0.7), (384, 192, 96, 48, 24),
                             (16, 24, 48, 120, 352)),
                            (896,  (1.2, 1.4, 0.7), (448, 224, 112, 56, 28),
                             (24, 32, 48, 136, 384)),
                            (1024, (1.4, 1.8, 0.6), (512, 256, 128, 64, 32),
                             (24, 32, 56, 160, 448)),
                            (1280, (1.6, 2.2, 0.6), (640, 320, 160, 80, 40),
                             (24, 40, 64, 176, 512)),
                            (1280, (1.8, 2.6, 0.5), (640, 320, 160, 80, 40),
                             (32, 40, 72, 200, 576)),
                            (1536, (1.8, 2.6, 0.5), (768, 384, 192, 96, 48),
                             (32, 40, 72, 200, 576))
                         ])
def test_EfficientNet_branch(input_shape, scaling_coefficients,
                             feature_shape, feature_channels):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    assert len(branch_tensors) == 5, "Number of features mismatch"
    for branch_tensor, feature_shape_per_tensor, feature_channel in zip(
            branch_tensors, feature_shape, feature_channels):
        target_shape = (None, feature_shape_per_tensor,
                        feature_shape_per_tensor, feature_channel)
        assert branch_tensor.shape == target_shape, (
            "Feature shape mismatch")
    del image, branch_tensors


@pytest.mark.parametrize(('input_shape, fusion'),
                         [
                            ((5, 5), 'fast'),
                            ((10, 10), 'sum'),
                            ((15, 10), 'fast'),
                            ((10, 15), 'sum'),
                            ((15, 25), 'fast'),
                            ((25, 15), 'sum'),
                            ((30, 25), 'fast'),
                            ((25, 30), 'sum')
                         ])
def test_fuse_feature(input_shape, fusion):
    x = tf.random.uniform(input_shape, minval=0, maxval=1,
                          dtype=tf.dtypes.float32)
    y = tf.random.uniform(input_shape, minval=0, maxval=1,
                          dtype=tf.dtypes.float32)
    z = tf.random.uniform(input_shape, minval=0, maxval=1,
                          dtype=tf.dtypes.float32)
    to_fuse = [x, y, z]
    fused_feature = FuseFeature(fusion=fusion)(to_fuse, fusion)
    assert fused_feature.shape == input_shape, 'Incorrect target shape'
    assert fused_feature.dtype == tf.dtypes.float32, (
        'Incorrect target datatype')
    del x, y, z


@pytest.mark.parametrize(('input_shape, scaling_coefficients, FPN_num_filters,'
                          'FPN_cell_repeats, fusion, output_shapes'),
                         [
                            (512,  (1.0, 1.0, 0.8), 64, 3, 'fast',
                                [(64, 64, 64), (32, 32, 64), (16, 16, 64),
                                 (8, 8, 64), (4, 4, 64)]),
                            (640,  (1.0, 1.1, 0.8), 88, 4, 'fast',
                                [(80, 80, 88), (40, 40, 88), (20, 20, 88),
                                 (10, 10, 88), (5, 5, 88)]),
                            (768,  (1.1, 1.2, 0.7), 112, 5, 'fast',
                                [(96, 96, 112), (48, 48, 112), (24, 24, 112),
                                 (12, 12, 112), (6, 6, 112)]),
                            (896,  (1.2, 1.4, 0.7), 160, 6, 'fast',
                                [(112, 112, 160), (56, 56, 160), (28, 28, 160),
                                 (14, 14, 160), (7, 7, 160)]),
                            (1024, (1.4, 1.8, 0.6), 224, 7, 'fast',
                                [(128, 128, 224), (64, 64, 224), (32, 32, 224),
                                 (16, 16, 224), (8, 8, 224)]),
                            (1280, (1.6, 2.2, 0.6), 288, 7, 'fast',
                                [(160, 160, 288), (80, 80, 288), (40, 40, 288),
                                 (20, 20, 288), (10, 10, 288)]),
                            (1280, (1.8, 2.6, 0.5), 384, 8, 'sum',
                                [(160, 160, 384), (80, 80, 384), (40, 40, 384),
                                 (20, 20, 384), (10, 10, 384)]),
                            (1536, (1.8, 2.6, 0.5), 384, 8, 'sum',
                                [(192, 192, 384), (96, 96, 384), (48, 48, 384),
                                 (24, 24, 384), (12, 12, 384)]),
                         ])
def test_EfficientDet_BiFPN(input_shape, scaling_coefficients, FPN_num_filters,
                            FPN_cell_repeats, fusion, output_shapes):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(
        branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    assert len(middles) == 5, "Incorrect middle features count"
    for middle, output_shape in zip(middles, output_shapes):
        target_shape = (None, ) + output_shape
        assert middle.shape == target_shape, "Middle feature shape mismatch"
    del branch_tensors, branches, middles, skips


@pytest.mark.parametrize(('input_shape, scaling_coefficients, FPN_num_filters,'
                          'FPN_cell_repeats, fusion, box_class_repeats,'
                          'output_shapes'),
                         [
                            (512,  (1.0, 1.0, 0.8), 64, 3, 'fast', 3,
                                (774144, 193536, 48384, 12096, 3024)),
                            (640,  (1.0, 1.1, 0.8), 88, 4, 'fast', 3,
                                (1209600, 302400, 75600, 18900, 4725)),
                            (768,  (1.1, 1.2, 0.7), 112, 5, 'fast', 3,
                                (1741824, 435456, 108864, 27216, 6804)),
                            (896,  (1.2, 1.4, 0.7), 160, 6, 'fast', 4,
                                (2370816, 592704, 148176, 37044, 9261)),
                            (1024, (1.4, 1.8, 0.6), 224, 7, 'fast', 4,
                                (3096576, 774144, 193536, 48384, 12096)),
                            (1280, (1.6, 2.2, 0.6), 288, 7, 'fast', 4,
                                (4838400, 1209600, 302400, 75600, 18900)),
                            (1280, (1.8, 2.6, 0.5), 384, 8, 'sum', 5,
                                (4838400, 1209600, 302400, 75600, 18900)),
                            (1536, (1.8, 2.6, 0.5), 384, 8, 'sum', 5,
                                (6967296, 1741824, 435456, 108864, 27216))
                         ])
def test_EfficientDet_ClassNet(input_shape, scaling_coefficients,
                               FPN_num_filters, FPN_cell_repeats, fusion,
                               box_class_repeats, output_shapes):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(
        branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    aspect_ratios = [1.0, 2.0, 0.5]
    num_scales = 3
    num_classes = 21
    survival_rate = None
    num_anchors = len(aspect_ratios) * num_scales
    args = (middles, num_anchors, FPN_num_filters,
            box_class_repeats, survival_rate)
    class_outputs = ClassNet(*args, num_classes)
    assert len(class_outputs) == 5, 'Class outputs length fail'
    for class_output, output_shape in zip(class_outputs, output_shapes):
        assert class_output.shape == (None, output_shape), (
            'Class outputs shape fail')
    del branch_tensors, branches, middles, skips, class_outputs


@pytest.mark.parametrize(('input_shape, scaling_coefficients, FPN_num_filters,'
                          'FPN_cell_repeats, fusion, box_class_repeats,'
                          'output_shapes'),
                         [
                            (512,  (1.0, 1.0, 0.8), 64, 3, 'fast', 3,
                                (147456, 36864, 9216, 2304, 576)),
                            (640,  (1.0, 1.1, 0.8), 88, 4, 'fast', 3,
                                (230400, 57600, 14400, 3600, 900)),
                            (768,  (1.1, 1.2, 0.7), 112, 5, 'fast', 3,
                                (331776, 82944, 20736, 5184, 1296)),
                            (896,  (1.2, 1.4, 0.7), 160, 6, 'fast', 4,
                                (451584, 112896, 28224, 7056, 1764)),
                            (1024, (1.4, 1.8, 0.6), 224, 7, 'fast', 4,
                                (589824, 147456, 36864, 9216, 2304)),
                            (1280, (1.6, 2.2, 0.6), 288, 7, 'fast', 4,
                                (921600, 230400, 57600, 14400, 3600)),
                            (1280, (1.8, 2.6, 0.5), 384, 8, 'sum', 5,
                                (921600, 230400, 57600, 14400, 3600)),
                            (1536, (1.8, 2.6, 0.5), 384, 8, 'sum', 5,
                                (1327104, 331776, 82944, 20736, 5184))
                         ])
def test_EfficientDet_BoxesNet(input_shape, scaling_coefficients,
                               FPN_num_filters, FPN_cell_repeats, fusion,
                               box_class_repeats, output_shapes):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(
        branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    aspect_ratios = [1.0, 2.0, 0.5]
    num_scales = 3
    num_dims = 4
    survival_rate = None
    num_anchors = len(aspect_ratios) * num_scales
    args = (middles, num_anchors, FPN_num_filters,
            box_class_repeats, survival_rate)
    boxes_outputs = BoxesNet(*args, num_dims)
    assert len(boxes_outputs) == 5
    for boxes_output, output_shape in zip(boxes_outputs, output_shapes):
        assert boxes_output.shape == (None, output_shape), (
            'Boxes outputs shape fail')
    del branch_tensors, branches, middles, skips, boxes_outputs


@pytest.mark.parametrize(('model, model_name, trainable_parameters,'
                          'non_trainable_parameters, input_shape,'
                          'output_shape'),
                         [
                            (EFFICIENTDETD0, 'efficientdet-d0', 3880067,
                                47136, (512, 512, 3), (49104, 94)),
                            (EFFICIENTDETD1, 'efficientdet-d1', 6625898,
                                71456, (640, 640, 3), (76725, 94)),
                            (EFFICIENTDETD2, 'efficientdet-d2', 8097039,
                                81776, (768, 768, 3), (110484, 94)),
                            (EFFICIENTDETD3, 'efficientdet-d3', 12032296,
                                114304, (896, 896, 3), (150381, 94)),
                            (EFFICIENTDETD4, 'efficientdet-d4', 20723675,
                                167312, (1024, 1024, 3), (196416, 94)),
                            (EFFICIENTDETD5, 'efficientdet-d5', 33653315,
                                227392, (1280, 1280, 3), (306900, 94)),
                            (EFFICIENTDETD6, 'efficientdet-d6', 51871934,
                                311984, (1280, 1280, 3), (306900, 94)),
                            (EFFICIENTDETD7, 'efficientdet-d7', 51871934,
                                311984, (1536, 1536, 3), (441936, 94)),
                         ])
def test_EfficientDet_architecture(model, model_name, model_input_name,
                                   model_output_name, trainable_parameters,
                                   non_trainable_parameters, input_shape,
                                   output_shape):
    implemented_model = model()
    trainable_count = count_params(
        implemented_model.trainable_weights)
    non_trainable_count = count_params(
        implemented_model.non_trainable_weights)
    assert implemented_model.name == model_name, "Model name incorrect"
    assert implemented_model.input_names[0] == model_input_name, (
        "Input name incorrect")
    assert implemented_model.output_names[0] == model_output_name, (
        "Output name incorrect")
    assert trainable_count == trainable_parameters, (
        "Incorrect trainable parameters count")
    assert non_trainable_count == non_trainable_parameters, (
        "Incorrect non-trainable parameters count")
    assert implemented_model.input_shape[1:] == input_shape, (
        "Incorrect input shape")
    assert implemented_model.output_shape[1:] == output_shape, (
        "Incorrect output shape")
    del implemented_model


@pytest.mark.parametrize(('model, image_size'),
                         [
                            (EFFICIENTDETD0, 512),
                            (EFFICIENTDETD1, 640),
                            (EFFICIENTDETD2, 768),
                            (EFFICIENTDETD3, 896),
                            (EFFICIENTDETD4, 1024),
                            (EFFICIENTDETD5, 1280),
                            (EFFICIENTDETD6, 1280),
                            (EFFICIENTDETD7, 1536),
                         ])
def test_EfficientDet_output(model, image_size):
    detector = model()
    image = get_test_images(image_size)
    output_shape = list(detector(image).shape)
    expected_output_shape = list(detector.prior_boxes.shape)
    num_classes = 90
    expected_output_shape[1] = expected_output_shape[1] + num_classes
    expected_output_shape = [1, ] + expected_output_shape
    assert output_shape == expected_output_shape, 'Outputs length fail'
    del detector


@pytest.mark.parametrize(('model, model_name'),
                         [
                            (EFFICIENTDETD0, 'efficientdet-d0'),
                            (EFFICIENTDETD1, 'efficientdet-d1'),
                            (EFFICIENTDETD2, 'efficientdet-d2'),
                            (EFFICIENTDETD3, 'efficientdet-d3'),
                            (EFFICIENTDETD4, 'efficientdet-d4'),
                            (EFFICIENTDETD5, 'efficientdet-d5'),
                            (EFFICIENTDETD6, 'efficientdet-d6'),
                            (EFFICIENTDETD7, 'efficientdet-d7'),
                         ])
def test_load_weights(model, model_name, model_weight_path):
    WEIGHT_PATH = model_weight_path
    base_weights = ['COCO', 'COCO']
    head_weights = ['COCO', None]
    num_classes = [90, 21]
    for base_weight, head_weight, num_class in zip(
            base_weights, head_weights, num_classes):
        detector = model(num_classes=num_class, base_weights=base_weight,
                         head_weights=head_weight)
        model_filename = '-'.join([model_name, base_weight, str(head_weight)
                                   + '_weights.hdf5'])
        weights_path = get_file(model_filename, WEIGHT_PATH + model_filename,
                                cache_subdir='paz/models')
        detector.load_weights(weights_path)
        del detector


@pytest.mark.parametrize(('model, aspect_ratios, num_boxes'),
                         [
                            (EFFICIENTDETD0, [1.0, 2.0, 0.5], 49104),
                            (EFFICIENTDETD1, [1.0, 2.0, 0.5], 76725),
                            (EFFICIENTDETD2, [1.0, 2.0, 0.5], 110484),
                            (EFFICIENTDETD3, [1.0, 2.0, 0.5], 150381),
                            (EFFICIENTDETD4, [1.0, 2.0, 0.5], 196416),
                            (EFFICIENTDETD5, [1.0, 2.0, 0.5], 306900),
                            (EFFICIENTDETD6, [1.0, 2.0, 0.5], 306900),
                            (EFFICIENTDETD7, [1.0, 2.0, 0.5], 441936),
                         ])
def test_prior_boxes(model, aspect_ratios, num_boxes):
    model = model()
    prior_boxes = model.prior_boxes
    anchor_x, anchor_y = prior_boxes[:, 0], prior_boxes[:, 1]
    anchor_W, anchor_H = prior_boxes[:, 2], prior_boxes[:, 3]
    measured_aspect_ratios = set(np.unique(np.round((anchor_W / anchor_H), 2)))
    assert np.logical_and(anchor_x >= 0, anchor_x <= 1).all(), (
        "Invalid x-coordinates of anchor centre")
    assert np.logical_and(anchor_y >= 0, anchor_y <= 1).all(), (
        "Invalid y-coordinates of anchor centre")
    assert (anchor_W > 0).all(), "Invalid/negative anchor width"
    assert (anchor_H > 0).all(), "Invalid/negative anchor height"
    assert np.round(np.mean(anchor_x), 2) == 0.5, (
        "Anchor boxes asymmetrically distributed along X-direction")
    assert np.round(np.mean(anchor_y), 2) == 0.5, (
        "Anchor boxes asymmetrically distributed along Y-direction")
    assert measured_aspect_ratios == set(aspect_ratios), (
        "Anchor aspect ratios not as expected")
    assert prior_boxes.shape[0] == num_boxes, (
        "Incorrect number of anchor boxes")
    del model


@pytest.mark.parametrize(('detection_pipeline, boxes_EFFICIENTDET'),
                         [
                            (EFFICIENTDETD0COCO, boxes_EFFICIENTDETD0COCO),
                            (EFFICIENTDETD1COCO, boxes_EFFICIENTDETD1COCO),
                            (EFFICIENTDETD2COCO, boxes_EFFICIENTDETD2COCO),
                            (EFFICIENTDETD3COCO, boxes_EFFICIENTDETD3COCO),
                            (EFFICIENTDETD4COCO, boxes_EFFICIENTDETD4COCO),
                            (EFFICIENTDETD5COCO, boxes_EFFICIENTDETD5COCO),
                            (EFFICIENTDETD6COCO, boxes_EFFICIENTDETD6COCO),
                            (EFFICIENTDETD7COCO, boxes_EFFICIENTDETD7COCO),
                         ])
def test_EfficientDet_inference(
        detection_pipeline, image_with_multiple_objects, boxes_EFFICIENTDET):
    detect = detection_pipeline(score_thresh=0.60, nms_thresh=0.25)
    detections = detect(image_with_multiple_objects)
    predicted_boxes2D = detections['boxes2D']
    labelled_boxes = boxes_EFFICIENTDET()
    assert len(predicted_boxes2D) == len(labelled_boxes)
    for box2D, predicted_box2D in zip(labelled_boxes, predicted_boxes2D):
        assert np.allclose(box2D.coordinates, predicted_box2D.coordinates)
        assert np.allclose(box2D.score, predicted_box2D.score)
        assert (box2D.class_name == predicted_box2D.class_name)
    del detection_pipeline
