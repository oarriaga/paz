import pytest
from keras.layers import Input, Flatten
from keras.models import Model
import numpy as np

from paz.models.detection.efficientdet.efficientnet import (
    EFFICIENTNET,
    apply_drop_connect,
    conv_block,
    MBconv_blocks,
    scale_filters,
    kernel_initializer,
    round_repeats,
    MB_repeat,
    MB_block,
)
from paz.models.detection.efficientdet.efficientdet_blocks import (
    ClassNet,
    BoxesNet,
    EfficientNet_to_BiFPN,
    BiFPN,
)


def test_conv_block():
    """Test the conv_block function."""
    input_tensor = Input(shape=(32, 32, 3))
    output_tensor = conv_block(
        input_tensor, intro_filters=[32], width_coefficient=1.0, depth_divisor=8
    )
    model = Model(inputs=input_tensor, outputs=output_tensor)
    # Expect spatial dimensions to be reduced (e.g., downsampled by 2)
    assert model.output_shape == (None, 16, 16, 32)


def test_MBconv_blocks():
    """Test the MBconv_blocks function."""
    input_tensor = Input(shape=(32, 32, 32))
    scaling_coefficients = (1.0, 1.0, 0.8)
    output_features = MBconv_blocks(
        input_tensor,
        kernel_sizes=[3],
        intro_filters=[32],
        outro_filters=[16],
        W_coefficient=scaling_coefficients[0],
        D_coefficient=scaling_coefficients[1],
        D_divisor=8,
        repeats=[1],
        excite_ratio=0.25,
        survival_rate=scaling_coefficients[2],
        strides=[[1, 1]],
        expand_ratios=[6],
    )
    assert len(output_features) == 1
    assert output_features[0].shape[-1] == 16


def test_efficientnet():
    """Test the EFFICIENTNET function."""
    input_tensor = Input(shape=(64, 64, 3))
    scaling_coefficients = (1.0, 1.0, 0.8)
    output_features = EFFICIENTNET(
        image=input_tensor, scaling_coefficients=scaling_coefficients
    )
    assert isinstance(output_features, list)
    assert len(output_features) == 5  # P3-P7 outputs


def test_scale_filters():
    """Test the scale_filters function."""
    filters = 32
    width_coefficient = 1.5
    depth_divisor = 8
    scaled_filters = scale_filters(filters, width_coefficient, depth_divisor)
    assert scaled_filters == 48


def test_kernel_initializer():
    """Test the kernel_initializer function."""
    shape = (3, 3, 3, 16)
    initializer = kernel_initializer(shape)
    assert initializer.shape == shape


def test_round_repeats():
    """Test the round_repeats function."""
    repeats = 2
    depth_coefficient = 1.5
    rounded_repeats = round_repeats(repeats, depth_coefficient)
    assert rounded_repeats == 3


def test_MB_repeat():
    """Test the MB_repeat function."""
    input_tensor = Input(shape=(32, 32, 32))
    output_tensor = MB_repeat(
        input_tensor,
        intro_filter=32,
        outro_filter=16,
        stride=[1, 1],
        repeats=1,
        block_args=(
            3,  # kernel_size
            0.8,  # survival_rate
            6,  # expand_ratio
            0.25,  # excite_ratio
        ),
    )
    model = Model(inputs=input_tensor, outputs=output_tensor)
    assert model.output_shape == (None, 32, 32, 16)


def test_MB_block():
    """Test the MB_block function."""
    input_tensor = Input(shape=(32, 32, 32))
    output_tensor = MB_block(
        input_tensor,
        intro_filters=32,
        outro_filters=16,
        strides=[1, 1],
        kernel_size=3,
        survival_rate=0.8,
        expand_ratio=6,
        excite_ratio=0.25,
    )
    model = Model(inputs=input_tensor, outputs=output_tensor)
    assert model.output_shape == (None, 32, 32, 16)


def test_apply_drop_connect():
    """Test the apply_drop_connect function."""

    # Create a dummy input tensor with a fixed batch size
    input_tensor = np.ones(shape=(4, 32, 32, 32))  # Batch size = 4
    output_tensor = apply_drop_connect(
        input_tensor, is_training=True, survival_rate=0.8
    )

    # Ensure the output tensor has the same shape as the input tensor
    assert output_tensor.shape == input_tensor.shape


@pytest.mark.parametrize(
    ("image_size, scaling_coefficients, output_shape"),
    [
        (512, (1.0, 1.0, 0.8), (256, 256, 16)),
        (640, (1.0, 1.1, 0.8), (320, 320, 16)),
        (768, (1.1, 1.2, 0.7), (384, 384, 16)),
        (896, (1.2, 1.4, 0.7), (448, 448, 24)),
        (1024, (1.4, 1.8, 0.6), (512, 512, 24)),
        (1280, (1.6, 2.2, 0.6), (640, 640, 24)),
        (1280, (1.8, 2.6, 0.5), (640, 640, 32)),
        (1536, (1.8, 2.6, 0.5), (768, 768, 32)),
    ],
)
def test_efficientnet_bottleneck_block(image_size, scaling_coefficients, output_shape):
    shape = (image_size, image_size, 3)
    image = Input(shape=shape, name="image")
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    assert (
        branch_tensors[0].shape == (None,) + output_shape
    ), "Bottleneck block output shape mismatch"


@pytest.mark.parametrize(
    ("input_shape, dtype, target_shape, is_training"),
    [
        ((1, 1), np.float64, (1, 1, 1, 1), True),
        ((5, 5), np.float64, (5, 1, 5, 5), True),
        ((3, 5), np.float64, (3, 1, 3, 5), True),
        ((5, 3), np.float64, (5, 1, 5, 3), True),
        ((1, 1), np.float64, (1, 1), False),
        ((5, 5), np.float64, (5, 5), False),
        ((3, 5), np.float64, (3, 5), False),
        ((5, 3), np.float64, (5, 3), False),
    ],
)
def test_drop_connect(input_shape, dtype, target_shape, is_training):
    # Generate random input using NumPy
    x = np.random.uniform(low=0, high=5, size=input_shape).astype(dtype)

    # Generate a random survival rate
    survival_rate = np.random.uniform(0.0, 1.0)

    # Apply drop connect (assumed to be implemented elsewhere)
    y = apply_drop_connect(x, is_training, survival_rate)

    # Validate the shape and dtype of the output
    assert y.shape == target_shape, "Incorrect target shape"
    assert y.dtype == dtype, "Incorrect target datatype"


@pytest.mark.parametrize(
    ("image_size, scaling_coefficients, output_shape"),
    [
        (512, (1.0, 1.0, 0.8), (256, 256, 16)),
        (640, (1.0, 1.1, 0.8), (320, 320, 16)),
        (768, (1.1, 1.2, 0.7), (384, 384, 16)),
        (896, (1.2, 1.4, 0.7), (448, 448, 24)),
        (1024, (1.4, 1.8, 0.6), (512, 512, 24)),
        (1280, (1.6, 2.2, 0.6), (640, 640, 24)),
        (1280, (1.8, 2.6, 0.5), (640, 640, 32)),
        (1536, (1.8, 2.6, 0.5), (768, 768, 32)),
    ],
)
def test_EfficientNet_SE_block(image_size, scaling_coefficients, output_shape):
    shape = (image_size, image_size, 3)
    image = Input(shape=shape, name="image")
    branch_tensors = EFFICIENTNET(image, scaling_coefficients, excite_ratio=0.8)
    assert (
        branch_tensors[0].shape == (None,) + output_shape
    ), "SE block output shape mismatch"
    del image, branch_tensors


@pytest.mark.parametrize(
    ("input_shape, scaling_coefficients, feature_shape," "feature_channels"),
    [
        (512, (1.0, 1.0, 0.8), (256, 128, 64, 32, 16), (16, 24, 40, 112, 320)),
        (640, (1.0, 1.1, 0.8), (320, 160, 80, 40, 20), (16, 24, 40, 112, 320)),
        (768, (1.1, 1.2, 0.7), (384, 192, 96, 48, 24), (16, 24, 48, 120, 352)),
        (896, (1.2, 1.4, 0.7), (448, 224, 112, 56, 28), (24, 32, 48, 136, 384)),
        (1024, (1.4, 1.8, 0.6), (512, 256, 128, 64, 32), (24, 32, 56, 160, 448)),
        (1280, (1.6, 2.2, 0.6), (640, 320, 160, 80, 40), (24, 40, 64, 176, 512)),
        (1280, (1.8, 2.6, 0.5), (640, 320, 160, 80, 40), (32, 40, 72, 200, 576)),
        (1536, (1.8, 2.6, 0.5), (768, 384, 192, 96, 48), (32, 40, 72, 200, 576)),
    ],
)
def test_EfficientNet_branch(
    input_shape, scaling_coefficients, feature_shape, feature_channels
):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name="image")
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    assert len(branch_tensors) == 5, "Number of features mismatch"
    for branch_tensor, feature_shape_per_tensor, feature_channel in zip(
        branch_tensors, feature_shape, feature_channels
    ):
        target_shape = (
            None,
            feature_shape_per_tensor,
            feature_shape_per_tensor,
            feature_channel,
        )
        assert branch_tensor.shape == target_shape, "Feature shape mismatch"
    del image, branch_tensors


@pytest.mark.parametrize(
    (
        "input_shape, scaling_coefficients, FPN_num_filters,"
        "FPN_cell_repeats, fusion, box_class_repeats,"
        "output_shapes"
    ),
    [
        (512, (1.0, 1.0, 0.8), 64, 3, "fast", 3, (774144, 193536, 48384, 12096, 3024)),
        (640, (1.0, 1.1, 0.8), 88, 4, "fast", 3, (1209600, 302400, 75600, 18900, 4725)),
        (
            768,
            (1.1, 1.2, 0.7),
            112,
            5,
            "fast",
            3,
            (1741824, 435456, 108864, 27216, 6804),
        ),
        (
            896,
            (1.2, 1.4, 0.7),
            160,
            6,
            "fast",
            4,
            (2370816, 592704, 148176, 37044, 9261),
        ),
        (
            1024,
            (1.4, 1.8, 0.6),
            224,
            7,
            "fast",
            4,
            (3096576, 774144, 193536, 48384, 12096),
        ),
        (
            1280,
            (1.6, 2.2, 0.6),
            288,
            7,
            "fast",
            4,
            (4838400, 1209600, 302400, 75600, 18900),
        ),
        (
            1280,
            (1.8, 2.6, 0.5),
            384,
            8,
            "sum",
            5,
            (4838400, 1209600, 302400, 75600, 18900),
        ),
        (
            1536,
            (1.8, 2.6, 0.5),
            384,
            8,
            "sum",
            5,
            (6967296, 1741824, 435456, 108864, 27216),
        ),
    ],
)
def test_EfficientDet_ClassNet(
    input_shape,
    scaling_coefficients,
    FPN_num_filters,
    FPN_cell_repeats,
    fusion,
    box_class_repeats,
    output_shapes,
):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name="image")
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    aspect_ratios = [1.0, 2.0, 0.5]
    num_scales = 3
    num_classes = 21
    survival_rate = None
    num_anchors = len(aspect_ratios) * num_scales
    args = (middles, num_anchors, FPN_num_filters, box_class_repeats, survival_rate)
    _, class_outputs = ClassNet(*args, num_classes)
    class_outputs = [Flatten()(class_output) for class_output in class_outputs]
    assert len(class_outputs) == 5, "Class outputs length fail"
    for class_output, output_shape in zip(class_outputs, output_shapes):
        assert class_output.shape == (None, output_shape), "Class outputs shape fail"
    del branch_tensors, branches, middles, skips, class_outputs


@pytest.mark.parametrize(
    (
        "input_shape, scaling_coefficients, FPN_num_filters,"
        "FPN_cell_repeats, fusion, box_class_repeats,"
        "output_shapes"
    ),
    [
        (512, (1.0, 1.0, 0.8), 64, 3, "fast", 3, (147456, 36864, 9216, 2304, 576)),
        (640, (1.0, 1.1, 0.8), 88, 4, "fast", 3, (230400, 57600, 14400, 3600, 900)),
        (768, (1.1, 1.2, 0.7), 112, 5, "fast", 3, (331776, 82944, 20736, 5184, 1296)),
        (896, (1.2, 1.4, 0.7), 160, 6, "fast", 4, (451584, 112896, 28224, 7056, 1764)),
        (1024, (1.4, 1.8, 0.6), 224, 7, "fast", 4, (589824, 147456, 36864, 9216, 2304)),
        (
            1280,
            (1.6, 2.2, 0.6),
            288,
            7,
            "fast",
            4,
            (921600, 230400, 57600, 14400, 3600),
        ),
        (1280, (1.8, 2.6, 0.5), 384, 8, "sum", 5, (921600, 230400, 57600, 14400, 3600)),
        (
            1536,
            (1.8, 2.6, 0.5),
            384,
            8,
            "sum",
            5,
            (1327104, 331776, 82944, 20736, 5184),
        ),
    ],
)
def test_EfficientDet_BoxesNet(
    input_shape,
    scaling_coefficients,
    FPN_num_filters,
    FPN_cell_repeats,
    fusion,
    box_class_repeats,
    output_shapes,
):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name="image")
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    aspect_ratios = [1.0, 2.0, 0.5]
    num_scales = 3
    num_dims = 4
    survival_rate = None
    num_anchors = len(aspect_ratios) * num_scales
    args = (middles, num_anchors, FPN_num_filters, box_class_repeats, survival_rate)
    _, boxes_outputs = BoxesNet(*args, num_dims)
    boxes_outputs = [Flatten()(boxes_output) for boxes_output in boxes_outputs]
    assert len(boxes_outputs) == 5
    for boxes_output, output_shape in zip(boxes_outputs, output_shapes):
        assert boxes_output.shape == (None, output_shape), "Boxes outputs shape fail"
    del branch_tensors, branches, middles, skips, boxes_outputs


@pytest.mark.parametrize(
    ("image_size, scaling_coefficients, output_shape"),
    [
        (512, (1.0, 1.0, 0.8), (1, 256, 256, 32)),
        (640, (1.0, 1.1, 0.8), (1, 320, 320, 32)),
        (768, (1.1, 1.2, 0.7), (1, 384, 384, 32)),
        (896, (1.2, 1.4, 0.7), (1, 448, 448, 40)),
        (1024, (1.4, 1.8, 0.6), (1, 512, 512, 48)),
        (1280, (1.6, 2.2, 0.6), (1, 640, 640, 48)),
        (1280, (1.8, 2.6, 0.5), (1, 640, 640, 56)),
        (1536, (1.8, 2.6, 0.5), (1, 768, 768, 56)),
    ],
)
def get_test_images(image_size, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)


def get_EfficientNet_hyperparameters():
    efficientnet_hyperparameters = {
        "intro_filters": [32, 16, 24, 40, 80, 112, 192],
        "outro_filters": [16, 24, 40, 80, 112, 192, 320],
        "D_divisor": 8,
        "kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
        "repeats": [1, 2, 2, 3, 3, 4, 1],
        "excite_ratio": 0.25,
        "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [1, 1], [2, 2], [1, 1]],
        "expand_ratios": [1, 6, 6, 6, 6, 6, 6],
    }
    return efficientnet_hyperparameters


@pytest.mark.parametrize(
    ("image_size, scaling_coefficients, output_shape"),
    [
        (512, (1.0, 1.0, 0.8), (1, 256, 256, 32)),
        (640, (1.0, 1.1, 0.8), (1, 320, 320, 32)),
        (768, (1.1, 1.2, 0.7), (1, 384, 384, 32)),
        (896, (1.2, 1.4, 0.7), (1, 448, 448, 40)),
        (1024, (1.4, 1.8, 0.6), (1, 512, 512, 48)),
        (1280, (1.6, 2.2, 0.6), (1, 640, 640, 48)),
        (1280, (1.8, 2.6, 0.5), (1, 640, 640, 56)),
        (1536, (1.8, 2.6, 0.5), (1, 768, 768, 56)),
    ],
)
def test_EfficientNet_conv_block(image_size, scaling_coefficients, output_shape):
    images = get_test_images(image_size, 1)
    efficientnet_hyperparameters = get_EfficientNet_hyperparameters()
    intro_filters = efficientnet_hyperparameters["intro_filters"]
    D_divisor = efficientnet_hyperparameters["D_divisor"]
    W_coefficient, D_coefficient, survival_rate = scaling_coefficients
    x = conv_block(images, intro_filters, W_coefficient, D_divisor)
    assert x.shape == output_shape, "Output shape mismatch"
    del images, x


@pytest.mark.parametrize(
    ("image_size, scaling_coefficients, output_shape"),
    [
        (
            512,
            (1.0, 1.0, 0.8),
            [
                (1, 256, 256, 16),
                (1, 128, 128, 24),
                (1, 64, 64, 40),
                (1, 32, 32, 112),
                (1, 16, 16, 320),
            ],
        ),
        (
            640,
            (1.0, 1.1, 0.8),
            [
                (1, 320, 320, 16),
                (1, 160, 160, 24),
                (1, 80, 80, 40),
                (1, 40, 40, 112),
                (1, 20, 20, 320),
            ],
        ),
        (
            768,
            (1.1, 1.2, 0.7),
            [
                (1, 384, 384, 16),
                (1, 192, 192, 24),
                (1, 96, 96, 48),
                (1, 48, 48, 120),
                (1, 24, 24, 352),
            ],
        ),
        (
            896,
            (1.2, 1.4, 0.7),
            [
                (1, 448, 448, 24),
                (1, 224, 224, 32),
                (1, 112, 112, 48),
                (1, 56, 56, 136),
                (1, 28, 28, 384),
            ],
        ),
        (
            1024,
            (1.4, 1.8, 0.6),
            [
                (1, 512, 512, 24),
                (1, 256, 256, 32),
                (1, 128, 128, 56),
                (1, 64, 64, 160),
                (1, 32, 32, 448),
            ],
        ),
        (
            1280,
            (1.6, 2.2, 0.6),
            [
                (1, 640, 640, 24),
                (1, 320, 320, 40),
                (1, 160, 160, 64),
                (1, 80, 80, 176),
                (1, 40, 40, 512),
            ],
        ),
        (
            1280,
            (1.8, 2.6, 0.5),
            [
                (1, 640, 640, 32),
                (1, 320, 320, 40),
                (1, 160, 160, 72),
                (1, 80, 80, 200),
                (1, 40, 40, 576),
            ],
        ),
        (
            1536,
            (1.8, 2.6, 0.5),
            [
                (1, 768, 768, 32),
                (1, 384, 384, 40),
                (1, 192, 192, 72),
                (1, 96, 96, 200),
                (1, 48, 48, 576),
            ],
        ),
    ],
)
def test_EfficientNet_MBconv_blocks(image_size, scaling_coefficients, output_shape):
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
        x,
        kernel_sizes,
        intro_filters,
        outro_filters,
        W_coefficient,
        D_coefficient,
        D_divisor,
        repeats,
        excite_ratio,
        survival_rate,
        strides,
        expand_ratios,
    )
    assert len(x) == len(output_shape), "Feature count mismatch"
    for feature, target_shape in zip(x, output_shape):
        assert feature.shape == target_shape, "Feature shape mismatch"
    del images, x


@pytest.mark.parametrize(
    (
        "input_shape, scaling_coefficients, FPN_num_filters,"
        "FPN_cell_repeats, fusion, output_shapes"
    ),
    [
        (
            512,
            (1.0, 1.0, 0.8),
            64,
            3,
            "fast",
            [(64, 64, 64), (32, 32, 64), (16, 16, 64), (8, 8, 64), (4, 4, 64)],
        ),
        (
            640,
            (1.0, 1.1, 0.8),
            88,
            4,
            "fast",
            [(80, 80, 88), (40, 40, 88), (20, 20, 88), (10, 10, 88), (5, 5, 88)],
        ),
        (
            768,
            (1.1, 1.2, 0.7),
            112,
            5,
            "fast",
            [(96, 96, 112), (48, 48, 112), (24, 24, 112), (12, 12, 112), (6, 6, 112)],
        ),
        (
            896,
            (1.2, 1.4, 0.7),
            160,
            6,
            "fast",
            [(112, 112, 160), (56, 56, 160), (28, 28, 160), (14, 14, 160), (7, 7, 160)],
        ),
        (
            1024,
            (1.4, 1.8, 0.6),
            224,
            7,
            "fast",
            [(128, 128, 224), (64, 64, 224), (32, 32, 224), (16, 16, 224), (8, 8, 224)],
        ),
        (
            1280,
            (1.6, 2.2, 0.6),
            288,
            7,
            "fast",
            [
                (160, 160, 288),
                (80, 80, 288),
                (40, 40, 288),
                (20, 20, 288),
                (10, 10, 288),
            ],
        ),
        (
            1280,
            (1.8, 2.6, 0.5),
            384,
            8,
            "sum",
            [
                (160, 160, 384),
                (80, 80, 384),
                (40, 40, 384),
                (20, 20, 384),
                (10, 10, 384),
            ],
        ),
        (
            1536,
            (1.8, 2.6, 0.5),
            384,
            8,
            "sum",
            [
                (192, 192, 384),
                (96, 96, 384),
                (48, 48, 384),
                (24, 24, 384),
                (12, 12, 384),
            ],
        ),
    ],
)
def test_EfficientDet_BiFPN(
    input_shape,
    scaling_coefficients,
    FPN_num_filters,
    FPN_cell_repeats,
    fusion,
    output_shapes,
):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name="image")
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    assert len(middles) == 5, "Incorrect middle features count"
    for middle, output_shape in zip(middles, output_shapes):
        target_shape = (None,) + output_shape
        assert middle.shape == target_shape, "Middle feature shape mismatch"
    del branch_tensors, branches, middles, skips


@pytest.mark.parametrize(
    (
        "input_shape, scaling_coefficients, FPN_num_filters,"
        "FPN_cell_repeats, fusion, box_class_repeats,"
        "output_shapes"
    ),
    [
        (512, (1.0, 1.0, 0.8), 64, 3, "fast", 3, (147456, 36864, 9216, 2304, 576)),
        (640, (1.0, 1.1, 0.8), 88, 4, "fast", 3, (230400, 57600, 14400, 3600, 900)),
        (768, (1.1, 1.2, 0.7), 112, 5, "fast", 3, (331776, 82944, 20736, 5184, 1296)),
        (896, (1.2, 1.4, 0.7), 160, 6, "fast", 4, (451584, 112896, 28224, 7056, 1764)),
        (1024, (1.4, 1.8, 0.6), 224, 7, "fast", 4, (589824, 147456, 36864, 9216, 2304)),
        (
            1280,
            (1.6, 2.2, 0.6),
            288,
            7,
            "fast",
            4,
            (921600, 230400, 57600, 14400, 3600),
        ),
        (1280, (1.8, 2.6, 0.5), 384, 8, "sum", 5, (921600, 230400, 57600, 14400, 3600)),
        (
            1536,
            (1.8, 2.6, 0.5),
            384,
            8,
            "sum",
            5,
            (1327104, 331776, 82944, 20736, 5184),
        ),
    ],
)
def test_EfficientDet_BoxesNet(
    input_shape,
    scaling_coefficients,
    FPN_num_filters,
    FPN_cell_repeats,
    fusion,
    box_class_repeats,
    output_shapes,
):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name="image")
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    aspect_ratios = [1.0, 2.0, 0.5]
    num_scales = 3
    num_dims = 4
    survival_rate = None
    num_anchors = len(aspect_ratios) * num_scales
    args = (middles, num_anchors, FPN_num_filters, box_class_repeats, survival_rate)
    _, boxes_outputs = BoxesNet(*args, num_dims)
    boxes_outputs = [Flatten()(boxes_output) for boxes_output in boxes_outputs]
    assert len(boxes_outputs) == 5
    for boxes_output, output_shape in zip(boxes_outputs, output_shapes):
        assert boxes_output.shape == (None, output_shape), "Boxes outputs shape fail"
    del branch_tensors, branches, middles, skips, boxes_outputs


if __name__ == "__main__":
    pytest.main([__file__])
