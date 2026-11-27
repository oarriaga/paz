import pytest
from paz.models.detection.single_shot_detector import build_prior_boxes
from paz.models.detection.single_shot_detector.ssd512 import SSD512


def test_untrained_model_predicts_one_row_per_prior_box():
    model = SSD512(num_classes=81, base_weights=None, head_weights=None)
    num_boxes = len(build_prior_boxes("COCO"))
    assert num_boxes == 24564
    assert model.output_shape == (None, num_boxes, 4 + 81)


def test_invalid_base_weights():
    with pytest.raises(ValueError, match="base_weights"):
        SSD512(base_weights="INVALID")


def test_invalid_head_weights():
    with pytest.raises(ValueError, match="head_weights"):
        SSD512(head_weights="INVALID")


def test_incompatible_num_classes_with_weights():
    with pytest.raises(ValueError, match="num_classes"):
        SSD512(num_classes=10, head_weights="COCO")


@pytest.mark.weights
@pytest.mark.parametrize(
    "num_classes, base_weights, head_weights",
    [
        (81, "COCO", "COCO"),
        (2, "OIV6Hand", "OIV6Hand"),
        (22, "COCO", "YCBVideo"),
        (2, "COCO", "OIV6Hand"),
    ],
)
def test_pretrained_weights_load(num_classes, base_weights, head_weights):
    model = SSD512(num_classes, base_weights, head_weights)
    assert model.output_shape == (None, 24564, 4 + num_classes)


@pytest.mark.parametrize(
    "num_classes, base_weights, head_weights",
    [
        (22, "OIV6Hand", "YCBVideo"),
        (2, "OIV6Hand", None),
        (81, "OIV6Hand", "COCO"),
        (81, None, "COCO"),
        (2, None, "OIV6Hand"),
        (22, None, "YCBVideo"),
    ],
)
def test_unsupported_weights_raise(num_classes, base_weights, head_weights):
    with pytest.raises(NotImplementedError):
        SSD512(num_classes, base_weights, head_weights)
