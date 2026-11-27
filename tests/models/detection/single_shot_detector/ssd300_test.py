import os
import h5py
import numpy as np
import pytest
from paz.models.detection.single_shot_detector import build_prior_boxes
from paz.models.detection.single_shot_detector.ssd300 import SSD300


def get_weights_path(filename):
    return os.path.expanduser("~/.keras/paz/models/" + filename)


def test_untrained_model_predicts_one_row_per_prior_box():
    model = SSD300(num_classes=21, base_weights=None, head_weights=None)
    num_boxes = len(build_prior_boxes("VOC"))
    assert num_boxes == 8732
    assert model.output_shape == (None, num_boxes, 4 + 21)


def test_invalid_base_weights():
    with pytest.raises(ValueError, match="base_weights"):
        SSD300(base_weights="INVALID")


def test_invalid_head_weights():
    with pytest.raises(ValueError, match="head_weights"):
        SSD300(head_weights="INVALID")


def test_incompatible_num_classes_with_weights():
    with pytest.raises(ValueError, match="num_classes"):
        SSD300(num_classes=10, head_weights="VOC")


@pytest.mark.weights
@pytest.mark.parametrize(
    "num_classes, base_weights, head_weights",
    [
        (21, "VOC", "VOC"),
        (22, "FAT", "FAT"),
        (21, "VGG", None),
        (21, "VOC", None),
    ],
)
def test_pretrained_weights_load(num_classes, base_weights, head_weights):
    model = SSD300(num_classes, base_weights, head_weights)
    assert model.output_shape == (None, 8732, 4 + num_classes)


@pytest.mark.parametrize(
    "num_classes, base_weights, head_weights",
    [
        (21, "VGG", "VOC"),
        (21, None, "VOC"),
        (22, "VGG", "FAT"),
        (22, None, "FAT"),
        (21, "FAT", "VOC"),
        (22, "VOC", "FAT"),
        (21, "FAT", None),
    ],
)
def test_unsupported_weights_raise(num_classes, base_weights, head_weights):
    with pytest.raises(NotImplementedError):
        SSD300(num_classes, base_weights, head_weights)


@pytest.mark.weights
def test_SSD300_VOC_None_loads_extra_layers_on_every_build():
    names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]
    models = [SSD300(21, "VOC", None) for _ in range(2)]
    with h5py.File(get_weights_path("SSD300-VOC-None_weights.hdf5")) as file:
        kernels = [file[name][name + "/kernel:0"][()] for name in names]
    for model in models:
        for name, kernel in zip(names, kernels):
            assert np.allclose(model.get_layer(name).kernel, kernel)
