import pytest
from keras.layers import Input
import pytest
from keras.layers import Input, Flatten
from keras.models import Model
import numpy as np
from keras.utils import get_file
import pytest
from keras.models import Model
import numpy as np

from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_detector_head,
    ClassNet,
    BoxesNet,
    EfficientNet_to_BiFPN,
    BiFPN,
)

from paz.models.detection.efficientdet.efficientdet import (
    EFFICIENTDETD0,
    EFFICIENTDETD1,
    EFFICIENTDETD2,
    EFFICIENTDETD3,
    EFFICIENTDETD4,
    EFFICIENTDETD5,
    EFFICIENTDETD6,
    EFFICIENTDETD7,
    EFFICIENTDETD7x,
)

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

from paz.models.detection.efficientdet.layers import GetDropConnect, FuseFeature


@pytest.fixture
def model_input_name():
    return "image"


@pytest.fixture
def model_output_name():
    return "boxes"


@pytest.fixture
def model_weight_path():
    WEIGHT_PATH = "https://github.com/oarriaga/altamira-data/releases/download/v0.16/"
    return WEIGHT_PATH


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
def get_test_images(image_size, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)


def test_build_detector_head():
    """Test the build_detector_head function."""
    input_tensors = [Input(shape=(16, 16, 64)) for _ in range(5)]
    output_tensor = build_detector_head(
        middles=input_tensors,
        num_classes=90,
        num_dims=4,
        aspect_ratios=[1.0, 2.0, 0.5],
        num_scales=3,
        FPN_num_filters=64,
        box_class_repeats=3,
        survival_rate=0.8,
    )
    # Last dimension should equal num_classes + num_dims (i.e., 90+4)
    assert output_tensor.shape[-1] == 94
    # Total anchors: grid size * num_anchors per cell * number of FPN levels
    expected_total_anchors = 16 * 16 * 9 * 5
    assert output_tensor.shape[1] == expected_total_anchors


def test_class_net():
    """Test the ClassNet function."""
    input_tensors = [Input(shape=(16, 16, 64)) for _ in range(5)]
    _, class_outputs = ClassNet(
        features=input_tensors,
        num_anchors=9,
        num_filters=32,
        num_blocks=4,
        survival_rate=0.8,
        num_classes=90,
    )
    assert len(class_outputs) == 5
    # Each output should have last dimension equal to num_classes * num_anchors
    assert class_outputs[0].shape[-1] == 90 * 9


def test_boxes_net():
    """Test the BoxesNet function."""
    input_tensors = [Input(shape=(16, 16, 64)) for _ in range(5)]
    _, boxes_outputs = BoxesNet(
        features=input_tensors,
        num_anchors=9,
        num_filters=32,
        num_blocks=4,
        survival_rate=0.8,
        num_dims=4,
    )
    assert len(boxes_outputs) == 5
    # Each output should have last dimension equal to num_dims * num_anchors
    assert boxes_outputs[0].shape[-1] == 4 * 9


def test_efficientnet_to_bifpn():
    """Test the EfficientNet_to_BiFPN function."""
    branches = [Input(shape=(16, 16, i * 32)) for i in range(1, 6)]
    branches, middles, skips = EfficientNet_to_BiFPN(branches, num_filters=64)
    assert len(middles) == 5
    assert len(skips) == 5
    # Check that the number of filters in the first middle branch is set correctly
    assert middles[0].shape[-1] == 64


def test_bifpn():
    """Test the BiFPN function."""
    middles = [Input(shape=(16 // (2**i), 16 // (2**i), 64)) for i in range(5)]
    skips = [
        None if i == 0 or i == 4 else Input(shape=(16 // (2**i), 16 // (2**i), 64))
        for i in range(5)
    ]
    new_middles, _ = BiFPN(middles, skips, num_filters=64, fusion="fast")
    assert len(new_middles) == 5
    assert new_middles[0].shape[-1] == 64


# Parameterize the test over all model functions.
@pytest.mark.parametrize(
    "model_fn",
    [
        EFFICIENTDETD0,
        EFFICIENTDETD1,
        EFFICIENTDETD2,
        EFFICIENTDETD3,
        EFFICIENTDETD4,
        EFFICIENTDETD5,
        EFFICIENTDETD6,
        EFFICIENTDETD7,
        EFFICIENTDETD7x,
    ],
)
def test_efficientdet_models(model_fn):
    """
    Test that each EfficientDet model can be created without downloading
    weights (by using base_weights=None, head_weights=None) and that the
    model has its prior boxes computed.
    """
    model = model_fn(num_classes=90, base_weights=None, head_weights=None)
    assert isinstance(model, Model)
    # Check that the model has a prior_boxes attribute that is not None.
    assert hasattr(model, "prior_boxes")
    assert model.prior_boxes is not None


# Parameterize the test over all model functions.
@pytest.mark.parametrize(
    "model_fn",
    [
        EFFICIENTDETD0,
        EFFICIENTDETD1,
        EFFICIENTDETD2,
        EFFICIENTDETD3,
        EFFICIENTDETD4,
        EFFICIENTDETD5,
        EFFICIENTDETD6,
        EFFICIENTDETD7,
        EFFICIENTDETD7x,
    ],
)
def count_params(weights):
    """Count the total number of scalars composing the weights."""
    unique_weights = {id(w): w for w in weights}.values()
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))


@pytest.mark.parametrize(
    (
        "model, model_name, trainable_parameters,"
        "non_trainable_parameters, input_shape,"
        "output_shape"
    ),
    [
        (EFFICIENTDETD0, "efficientdet-d0", 3880067, 47136, (512, 512, 3), (49104, 94)),
        (EFFICIENTDETD1, "efficientdet-d1", 6625898, 71456, (640, 640, 3), (76725, 94)),
        (
            EFFICIENTDETD2,
            "efficientdet-d2",
            8097039,
            81776,
            (768, 768, 3),
            (110484, 94),
        ),
        (
            EFFICIENTDETD3,
            "efficientdet-d3",
            12032296,
            114304,
            (896, 896, 3),
            (150381, 94),
        ),
        (
            EFFICIENTDETD4,
            "efficientdet-d4",
            20723675,
            167312,
            (1024, 1024, 3),
            (196416, 94),
        ),
        (
            EFFICIENTDETD5,
            "efficientdet-d5",
            33653315,
            227392,
            (1280, 1280, 3),
            (306900, 94),
        ),
        (
            EFFICIENTDETD6,
            "efficientdet-d6",
            51871934,
            311984,
            (1280, 1280, 3),
            (306900, 94),
        ),
        (
            EFFICIENTDETD7,
            "efficientdet-d7",
            51871934,
            311984,
            (1536, 1536, 3),
            (441936, 94),
        ),
    ],
)
def test_efficientdet_architecture(
    model,
    model_name,
    trainable_parameters,
    non_trainable_parameters,
    input_shape,
    output_shape,
):
    implemented_model = model()
    trainable_count = count_params(implemented_model.trainable_weights)
    non_trainable_count = count_params(implemented_model.non_trainable_weights)
    assert implemented_model.name == model_name, "Model name incorrect"
    assert implemented_model.input_shape[1:] == input_shape, "Incorrect input shape"
    assert implemented_model.output_shape[1:] == output_shape, "Incorrect output shape"
    assert (
        trainable_count == trainable_parameters
    ), "Incorrect trainable parameters count"
    assert (
        non_trainable_count == non_trainable_parameters
    ), "Incorrect non-trainable parameters count"


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


def test_get_drop_connect():
    """Test the GetDropConnect layer."""
    layer = GetDropConnect(survival_rate=0.8)
    input_tensor = np.random.rand(1, 4, 4, 3).astype("float32")

    # Test training mode (dropout applied)
    output_train = layer.call(input_tensor, training=True)
    assert output_train.shape == (1, 4, 4, 3)
    # Values should change in training mode due to dropout
    assert not np.allclose(output_train, input_tensor)

    # Test inference mode (no dropout)
    output_inference = layer.call(input_tensor, training=False)
    assert np.allclose(output_inference, input_tensor)


def test_fuse_feature():
    """Test the FuseFeature layer."""
    # Test 'fast' fusion
    layer_fast = FuseFeature(fusion="fast")
    input_shape = [(1, 4, 4, 3)] * 3
    layer_fast.build(input_shape)
    input_tensors = [np.random.rand(1, 4, 4, 3).astype("float32") for _ in range(3)]
    output_fast = layer_fast.call(input_tensors, fusion="fast")
    assert output_fast.shape == (1, 4, 4, 3)

    # Test 'sum' fusion
    layer_sum = FuseFeature(fusion="sum")
    output_sum = layer_sum.call(input_tensors, fusion="sum")
    assert output_sum.shape == (1, 4, 4, 3)
    expected_sum = input_tensors[0] + input_tensors[1] + input_tensors[2]
    assert np.allclose(output_sum, expected_sum)


@pytest.mark.parametrize(
    ("model, image_size"),
    [
        (EFFICIENTDETD0, 512),
        (EFFICIENTDETD1, 640),
        (EFFICIENTDETD2, 768),
        (EFFICIENTDETD3, 896),
        (EFFICIENTDETD4, 1024),
        (EFFICIENTDETD5, 1280),
        (EFFICIENTDETD6, 1280),
        (EFFICIENTDETD7, 1536),
    ],
)
def test_EfficientDet_output(model, image_size):
    detector = model()
    image = get_test_images(image_size)
    output_shape = list(detector(image).shape)
    expected_output_shape = list(detector.prior_boxes.shape)
    num_classes = 90
    expected_output_shape[1] = expected_output_shape[1] + num_classes
    expected_output_shape = [
        1,
    ] + expected_output_shape
    assert output_shape == expected_output_shape, "Outputs length fail"
    del detector


@pytest.mark.parametrize(
    ("model, model_name"),
    [
        (EFFICIENTDETD0, "efficientdet-d0"),
        (EFFICIENTDETD1, "efficientdet-d1"),
        (EFFICIENTDETD2, "efficientdet-d2"),
        (EFFICIENTDETD3, "efficientdet-d3"),
        (EFFICIENTDETD4, "efficientdet-d4"),
        (EFFICIENTDETD5, "efficientdet-d5"),
        (EFFICIENTDETD6, "efficientdet-d6"),
        (EFFICIENTDETD7, "efficientdet-d7"),
    ],
)
def test_load_weights(model, model_name, model_weight_path):
    WEIGHT_PATH = model_weight_path
    base_weights = ["COCO", "COCO"]
    head_weights = ["COCO", None]
    num_classes = [90, 21]
    for base_weight, head_weight, num_class in zip(
        base_weights, head_weights, num_classes
    ):
        detector = model(
            num_classes=num_class, base_weights=base_weight, head_weights=head_weight
        )
        model_filename = "-".join(
            [model_name, base_weight, str(head_weight) + "_weights.hdf5"]
        )
        weights_path = get_file(
            model_filename, WEIGHT_PATH + model_filename, cache_subdir="paz/models"
        )
        detector.load_weights(weights_path)
        del detector


@pytest.mark.parametrize(
    ("model, aspect_ratios, num_boxes"),
    [
        (EFFICIENTDETD0, [1.0, 2.0, 0.5], 49104),
        (EFFICIENTDETD1, [1.0, 2.0, 0.5], 76725),
        (EFFICIENTDETD2, [1.0, 2.0, 0.5], 110484),
        (EFFICIENTDETD3, [1.0, 2.0, 0.5], 150381),
        (EFFICIENTDETD4, [1.0, 2.0, 0.5], 196416),
        (EFFICIENTDETD5, [1.0, 2.0, 0.5], 306900),
        (EFFICIENTDETD6, [1.0, 2.0, 0.5], 306900),
        (EFFICIENTDETD7, [1.0, 2.0, 0.5], 441936),
    ],
)
def test_prior_boxes(model, aspect_ratios, num_boxes):
    model = model()
    prior_boxes = model.prior_boxes
    anchor_x, anchor_y = prior_boxes[:, 0], prior_boxes[:, 1]
    anchor_W, anchor_H = prior_boxes[:, 2], prior_boxes[:, 3]
    measured_aspect_ratios = set(np.unique(np.round((anchor_W / anchor_H), 2)))
    assert np.logical_and(
        anchor_x >= 0, anchor_x <= 1
    ).all(), "Invalid x-coordinates of anchor centre"
    assert np.logical_and(
        anchor_y >= 0, anchor_y <= 1
    ).all(), "Invalid y-coordinates of anchor centre"
    assert (anchor_W > 0).all(), "Invalid/negative anchor width"
    assert (anchor_H > 0).all(), "Invalid/negative anchor height"
    assert (
        np.round(np.mean(anchor_x), 2) == 0.5
    ), "Anchor boxes asymmetrically distributed along X-direction"
    assert (
        np.round(np.mean(anchor_y), 2) == 0.5
    ), "Anchor boxes asymmetrically distributed along Y-direction"
    assert measured_aspect_ratios == set(
        aspect_ratios
    ), "Anchor aspect ratios not as expected"
    assert prior_boxes.shape[0] == num_boxes, "Incorrect number of anchor boxes"
    del model


if __name__ == "__main__":
    pytest.main([__file__])
