import jax
import keras
import numpy as np
from keras import Model

from paz.applications.detectors import PFRPN
from paz.models.detection.pf_rpn import decoder, swin
from paz.models.detection.pf_rpn.models import build_pf_rpn

IMAGE_SHAPE = (192, 256, 3)
NUM_QUERIES = 20


def build_model():
    """A narrow, shallow model so the tests stay quick; seeded to repeat."""
    keras.utils.set_random_seed(0)
    args = 32, (2, 2, 2, 2), (1, 2, 4, 8), 4, 1, 2, NUM_QUERIES
    return build_pf_rpn(IMAGE_SHAPE, *args, "pf_rpn_test")


def make_input(batch=2):
    random = np.random.RandomState(0)
    return random.randn(batch, *IMAGE_SHAPE).astype("float32")


def test_returns_scores_and_boxes():
    scores, boxes = build_model()(make_input())
    assert tuple(scores.shape) == (2, NUM_QUERIES)
    assert tuple(boxes.shape) == (2, NUM_QUERIES, 4)
    assert np.all(np.isfinite(scores)) and np.all(np.isfinite(boxes))


def test_neck_jit_matches_eager():
    """Compares up to the neck: untrained scores are near tied, so a float32
    difference can reorder the selected queries and every later tensor."""
    model = build_model()
    neck = Model(model.input, model.get_layer("neck_extra_convs_0_gn").output)
    eager = np.array(neck(make_input()))
    jitted = np.array(jax.jit(lambda x: neck(x))(make_input()))
    assert np.allclose(eager, jitted, atol=1e-4)


def test_compiles_once_for_any_image_size():
    model = build_model()
    forward = jax.jit(lambda x: model(x))
    propose = PFRPN(forward, IMAGE_SHAPE[:2], 0.5, 10, None)
    random = np.random.RandomState(0)
    for size in ((120, 160), (300, 200), (192, 256)):
        boxes, scores = propose(random.randint(0, 255, (*size, 3), "uint8"))
        assert boxes.shape[1] == 4 and len(boxes) == len(scores)
    assert forward._cache_size() == 1


def test_selected_cells_do_not_repeat():
    random = np.random.RandomState(0)
    confidence, similarity = random.rand(2, 1, 50).astype("float32")
    selected = np.array(decoder.select_cells(confidence, similarity, 20))
    assert len(np.unique(selected)) == 20
    assert set(np.argsort(-confidence[0])[:10]) == set(selected[0, :10])


def test_shift_mask_blocks_only_wrapped_cells():
    assert np.all(swin.build_shift_mask(8, 8, 4, 0) == 0.0)
    mask = swin.build_shift_mask(8, 8, 4, 2)
    assert mask.shape == (4, 1, 16, 16)
    assert np.all(mask[0] == 0.0) and np.any(mask[3] == -100.0)
