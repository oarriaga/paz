import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax

from paz.losses.pose import MultiPoseLoss


LINEMOD_CAMERA = np.array([[572.4, 0.0, 325.2],
                           [0.0, 573.5, 242.0],
                           [0.0, 0.0, 1.0]], dtype="float32")


def build_inputs(num_boxes=200, num_points=40, num_positives=4, seed=0):
    rng = np.random.default_rng(seed)
    priors = np.zeros((num_boxes, 3), dtype="float32")
    priors[:, 0] = rng.uniform(0, 512, num_boxes)
    priors[:, 1] = rng.uniform(0, 512, num_boxes)
    priors[:, 2] = 8.0
    model_points = rng.normal(size=(num_points, 3)).astype("float32")
    y_true = np.zeros((1, num_boxes, 11), dtype="float32")
    y_true[0, :num_positives, 0:3] = rng.normal(size=(num_positives, 3)) * 0.2
    y_true[0, :num_positives, 6:9] = rng.normal(size=(num_positives, 3)) * 0.1
    y_true[0, :num_positives, -2] = 1.0
    y_true[0, :, -1] = 1.0
    return priors, model_points, y_true


def make_loss(priors, model_points):
    return MultiPoseLoss(model_points, priors, LINEMOD_CAMERA, max_positives=8)


def test_pose_loss_is_finite_scalar():
    priors, points, y_true = build_inputs()
    loss = make_loss(priors, points)
    y_pred = np.zeros((1, y_true.shape[1], 6), dtype="float32")
    value = float(loss.compute_loss(y_true, y_pred))
    assert np.isfinite(value) and value >= 0.0


def test_pose_loss_zero_with_no_positives():
    priors, points, y_true = build_inputs(num_positives=0)
    loss = make_loss(priors, points)
    y_pred = np.zeros((1, y_true.shape[1], 6), dtype="float32")
    assert float(loss.compute_loss(y_true, y_pred)) == 0.0


def test_pose_loss_increases_with_rotation_error():
    priors, points, y_true = build_inputs()
    loss = make_loss(priors, points)
    y_pred = np.zeros((1, y_true.shape[1], 6), dtype="float32")
    y_pred[0, :, 0:3] = y_true[0, :, 0:3]
    small = float(loss.compute_loss(y_true, y_pred))
    y_pred[0, :4, 0:3] = y_true[0, :4, 0:3] + 0.5
    large = float(loss.compute_loss(y_true, y_pred))
    assert large > small


def test_pose_loss_is_differentiable():
    priors, points, y_true = build_inputs()
    loss = make_loss(priors, points)
    rng = np.random.default_rng(1)
    y_pred = rng.normal(size=(1, y_true.shape[1], 6)).astype("float32") * 0.1

    def scalar(y_pred):
        return loss.compute_loss(y_true, y_pred)

    gradient = np.asarray(jax.grad(scalar)(y_pred))
    assert np.isfinite(gradient).all() and np.abs(gradient).sum() > 0.0
