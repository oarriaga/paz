import jax.numpy as jp
import numpy as np

import paz
from paz.applications.detectors import EfficientDet


def build_fake_efficientdet(shifts):
    """Predicts one confident 102 pixel box per `(x, y)` anchor shift."""
    rows = []
    for class_arg, (shift_x, shift_y) in enumerate(shifts):
        scores = np.full(90, 1e-4)
        scores[class_arg] = 0.99
        rows.append(np.concatenate([[shift_y, shift_x, 0.0, 0.0], scores]))
    predictions = jp.array(rows, "float32")[None]

    def model(image):
        return predictions

    model.input_shape = (None, 512, 512, 3)
    model.prior_boxes = np.tile([[0.5, 0.5, 0.2, 0.2]], (len(shifts), 1))
    return model


def test_EfficientDet_keeps_boxes_cut_by_image_border():
    centered, cut_by_left, cut_by_bottom = (0.0, 0.0), (-2.6, 0.0), (0.0, 2.6)
    model = build_fake_efficientdet([centered, cut_by_left, cut_by_bottom])
    names = paz.datasets.labels("COCO_EFFICIENTDET")
    detect = EfficientDet(model, names, top_k=3, draw=False)
    boxes, labels, scores = detect(np.zeros((512, 512, 3), "uint8"))
    assert sorted(labels.tolist()) == [0, 1, 2]
    assert boxes.min() >= 0
    assert boxes.max() <= 511
