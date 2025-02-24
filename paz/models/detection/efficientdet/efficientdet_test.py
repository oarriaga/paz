import pytest
from keras.models import Model
import numpy as np
from keras.utils import get_file

# Import the functions to be tested
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


def get_test_images(image_size, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)


@pytest.fixture
def model_weight_path():
    WEIGHT_PATH = "https://github.com/oarriaga/altamira-data/releases/download/v0.16/"
    return WEIGHT_PATH


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
