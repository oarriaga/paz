import os

import numpy as np
import jax
import keras
from keras import Model

from paz.models.detection.rf_detr import models
from paz.models.detection.rf_detr.models import build_rf_detr
from paz.models.detection.rf_detr.models import build_trainable_rf_detr
from paz.models.detection.rf_detr.models import NUM_QUERIES

IMAGE_SHAPE = (288, 288, 3)
NUM_CLASSES = 7


def build_model():
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
    logits, boxes = build_model()(make_input())
    assert tuple(logits.shape) == (2, NUM_QUERIES, NUM_CLASSES)
    assert tuple(boxes.shape) == (2, NUM_QUERIES, 4)


def test_output_is_two_plain_tensors():
    output = build_model()(make_input())
    assert not isinstance(output, dict)
    assert not hasattr(output, "_fields")
    assert len(output) == 2


def test_boxes_have_positive_sizes():
    boxes = np.array(build_model()(make_input())[1])
    assert np.all(boxes[..., 2:] > 0.0)
    assert np.all(np.isfinite(boxes))


def test_backbone_jit_matches_eager():
    """Compares the graph up to the projector, before any query ranking.

    Untrained first-stage scores are near tied, so a float32 difference of
    1e-5 can reorder the selected queries and change every later tensor. The
    ranking is stable on trained weights, which parity_test.py checks.
    """
    model = build_model()
    features = Model(model.input, model.get_layer("projector_norm").output)
    data = make_input()
    eager = np.array(features(data))
    jitted = np.array(jax.jit(lambda x: features(x))(data))
    assert np.allclose(eager, jitted, atol=1e-4)


def test_untapped_blocks_are_pruned():
    names = {layer.name for layer in build_model().layers}
    assert "block_2_norm1" in names
    assert "block_3_norm1" not in names


def build_trainable(num_groups):
    keras.utils.set_random_seed(0)
    args = IMAGE_SHAPE, 16, 2, (1,), (0, 2), 2
    args = args + (NUM_CLASSES, num_groups, "rf_detr_test")
    return build_trainable_rf_detr(*args)


def test_stacks_the_first_stage_and_every_decoder_layer():
    output = build_trainable(1)(make_input())
    columns = 4 + NUM_CLASSES
    assert tuple(output.shape) == (2, 3, 1, NUM_QUERIES, columns)


def test_query_groups_add_a_group_axis():
    output = build_trainable(3)(make_input())
    columns = 4 + NUM_CLASSES
    assert tuple(output.shape) == (2, 3, 3, NUM_QUERIES, columns)


def read_weighted_names(model):
    """Names of the layers that own weights, which is what a load matches."""
    return {layer.name for layer in model.layers if layer.weights}


def test_group_zero_keeps_the_ungrouped_names():
    grouped = read_weighted_names(build_trainable(2))
    assert read_weighted_names(build_trainable(1)).issubset(grouped)
    assert read_weighted_names(build_model()).issubset(grouped)
    assert "enc_output_group_1" in grouped
    assert "query_feat_group_1" in grouped


def test_detector_matches_the_last_stage():
    model = build_trainable(2)
    data = make_input()
    stages = np.array(model(data))
    logits, boxes = models.build_detector(model)(data)
    assert np.allclose(np.array(logits), stages[:, -1, 0, :, 4:], atol=1e-6)
    assert np.allclose(np.array(boxes), stages[:, -1, 0, :, :4], atol=1e-6)


def test_query_groups_predict_different_boxes():
    """Each group owns its tables, so they must not collapse into one."""
    boxes = np.array(build_trainable(2)(make_input()))[:, -1, :, :, :4]
    assert not np.allclose(boxes[:, 0], boxes[:, 1], atol=1e-4)


def test_self_attention_stays_inside_a_group():
    model = build_trainable(2)
    data = make_input()
    before = np.array(model(data))[:, -1, 0]
    table = model.get_layer("query_feat_group_1")
    table.set_weights([100.0 * weight for weight in table.get_weights()])
    after = np.array(model(data))[:, -1, 0]
    assert np.allclose(before, after, atol=1e-4)


def copy_weights(source, model):
    for layer in source.layers:
        if layer.weights:
            model.get_layer(layer.name).set_weights(layer.get_weights())


def test_grouped_layer_names_stay_unique():
    """A bare group index would collide with the box head suffixes."""
    names = [layer.name for layer in build_trainable(3).layers]
    assert len(names) == len(set(names))


def test_added_groups_leave_group_zero_alone():
    """Grouping folds groups into the batch, which retiles the matmuls, so
    the same weights come back within float32 noise rather than exactly."""
    single, grouped = build_trainable(1), build_trainable(3)
    copy_weights(single, grouped)
    data = make_input()
    expected = np.array(single(data))[:, :, 0]
    assert np.allclose(np.array(grouped(data))[:, :, 0], expected, atol=1e-3)


def test_a_trainable_checkpoint_loads_into_a_detector(tmp_path):
    """Keras keys a weights file by layer class and position, not by name,
    so both graphs have to keep their weighted layers in the same order."""
    trainable = build_trainable(1)
    path = os.path.join(str(tmp_path), "trainable.weights.h5")
    trainable.save_weights(path)
    data = make_input()
    expected = np.array(trainable(data))[:, -1, 0]
    model = build_model()
    model.load_weights(path)
    logits, boxes = model(data)
    assert np.allclose(np.array(logits), expected[..., 4:], atol=1e-6)
    assert np.allclose(np.array(boxes), expected[..., :4], atol=1e-6)
