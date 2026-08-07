import numpy as np
import jax
import jax.numpy as jp

from paz.losses import detr

NUM_QUERIES, NUM_CLASSES, MAX_BOXES = 6, 3, 2


def build_targets(boxes_and_labels):
    padding = [[-1.0] * 5] * (MAX_BOXES - len(boxes_and_labels))
    return jp.asarray([boxes_and_labels + padding], "float32")


def build_predictions(boxes, labels, confidence=8.0):
    logits = np.full((1, NUM_QUERIES, NUM_CLASSES), -confidence, "float32")
    predicted = np.full((1, NUM_QUERIES, 4), 0.5, "float32")
    for query, (box, label) in enumerate(zip(boxes, labels)):
        predicted[0, query] = box
        logits[0, query, label] = confidence
    return jp.concatenate([jp.asarray(predicted), jp.asarray(logits)], axis=-1)


def test_perfect_prediction_has_no_box_error():
    box = [0.5, 0.5, 0.2, 0.2]
    y_true = build_targets([box + [1.0]])
    y_pred = build_predictions([box], [1])
    assert float(detr.regression(y_true, y_pred)) == 0.0
    assert float(detr.generalized_IOU(y_true, y_pred)) < 1e-5


def test_matching_ignores_query_order():
    box_0, box_1 = [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]
    y_true = build_targets([box_0 + [0.0], box_1 + [2.0]])
    forward = build_predictions([box_0, box_1], [0, 2])
    reversed_order = build_predictions([box_1, box_0], [2, 0])
    assert np.allclose(float(detr.call(y_true, forward)),
                       float(detr.call(y_true, reversed_order)), atol=1e-5)


def test_padding_values_do_not_contribute():
    """Only the class column marks padding, so its box values must not count."""
    box = [0.5, 0.5, 0.2, 0.2]
    y_pred = build_predictions([box], [1])
    padded = jp.asarray([[box + [1.0], [-1.0] * 5]], "float32")
    scrambled = jp.asarray([[box + [1.0], [0.3, 0.7, 0.4, 0.4, -1.0]]])
    expected = float(detr.call(y_true=padded, y_pred=y_pred))
    actual = float(detr.call(y_true=jp.asarray(scrambled, "float32"),
                             y_pred=y_pred))
    assert np.allclose(expected, actual, atol=1e-6)


def test_wrong_boxes_cost_more_than_right_ones():
    box = [0.5, 0.5, 0.2, 0.2]
    y_true = build_targets([box + [1.0]])
    right = detr.call(y_true, build_predictions([box], [1]))
    wrong = detr.call(y_true, build_predictions([[0.1, 0.1, 0.2, 0.2]], [1]))
    assert float(wrong) > float(right)


def test_wrong_label_costs_more_than_right_one():
    box = [0.5, 0.5, 0.2, 0.2]
    y_true = build_targets([box + [1.0]])
    right = detr.classification(y_true, build_predictions([box], [1]))
    wrong = detr.classification(y_true, build_predictions([box], [0]))
    assert float(wrong) > float(right)


def test_gradient_flows_into_predictions():
    box = [0.5, 0.5, 0.2, 0.2]
    y_true = build_targets([box + [1.0]])
    y_pred = build_predictions([[0.4, 0.4, 0.3, 0.3]], [1])
    gradient = jax.grad(lambda x: detr.call(y_true, x))(y_pred)
    assert np.all(np.isfinite(np.array(gradient)))
    assert np.abs(np.array(gradient)).max() > 0.0


def test_call_is_jit_compatible():
    box = [0.5, 0.5, 0.2, 0.2]
    y_true = build_targets([box + [1.0]])
    y_pred = build_predictions([box], [1])
    eager = float(detr.call(y_true, y_pred))
    jitted = float(jax.jit(detr.call)(y_true, y_pred))
    assert np.allclose(eager, jitted, atol=1e-5)
