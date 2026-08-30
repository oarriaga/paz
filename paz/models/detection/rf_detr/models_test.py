import numpy as np
import jax
import keras
from keras import Model

from paz.models.detection.rf_detr.models import build_rf_detr
from paz.models.detection.rf_detr.models import NUM_QUERIES

IMAGE_SHAPE = (288, 288, 3)
NUM_CLASSES = 7


def build_detector():
    """Taps early blocks so Keras prunes the rest and the test stays quick.

    The grid holds 18 x 18 = 324 cells, enough for the 300 queries the first
    stage selects from it. Seeded so the numerical assertions below compare the
    same weights on every run.
    """
    keras.utils.set_random_seed(0)
    args = IMAGE_SHAPE, 16, 2, (1,), (0, 2), 2
    return build_rf_detr(*args, NUM_CLASSES, "rf_detr_test")


def make_input(batch=2):
    random = np.random.RandomState(0)
    return random.randn(batch, *IMAGE_SHAPE).astype("float32")


def test_returns_logits_and_boxes():
    logits, boxes = build_detector()(make_input())
    assert tuple(logits.shape) == (2, NUM_QUERIES, NUM_CLASSES)
    assert tuple(boxes.shape) == (2, NUM_QUERIES, 4)


def test_output_is_two_plain_tensors():
    output = build_detector()(make_input())
    assert not isinstance(output, dict)
    assert not hasattr(output, "_fields")
    assert len(output) == 2


def test_boxes_have_positive_sizes():
    boxes = np.array(build_detector()(make_input())[1])
    assert np.all(boxes[..., 2:] > 0.0)
    assert np.all(np.isfinite(boxes))


def test_backbone_jit_matches_eager():
    """Compares the graph up to the projector, before any query ranking.

    Untrained first-stage scores are near tied, so a float32 difference of
    1e-5 can reorder the selected queries and change every later tensor. The
    ranking is stable on trained weights, which parity_test.py checks.
    """
    model = build_detector()
    features = Model(model.input, model.get_layer("projector_norm").output)
    data = make_input()
    eager = np.array(features(data))
    jitted = np.array(jax.jit(lambda x: features(x))(data))
    assert np.allclose(eager, jitted, atol=1e-4)


def test_untapped_blocks_are_pruned():
    names = {layer.name for layer in build_detector().layers}
    assert "block_2_norm1" in names
    assert "block_3_norm1" not in names
