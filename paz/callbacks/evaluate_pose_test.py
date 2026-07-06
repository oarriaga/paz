import numpy as np
from paz.callbacks import EvaluatePose


def unit_points(seed=0):
    return np.random.RandomState(seed).uniform(-1.0, 1.0, (40, 3))


def test_evaluate_pose_perfect_prediction():
    points3D = unit_points()
    poses_true = [(np.eye(3), np.array([0.0, 0.0, 1.0]))]

    def predict_pose(model, image):
        return (np.eye(3), np.array([0.0, 0.0, 1.0]))

    callback = EvaluatePose([np.zeros((4, 4, 3))], poses_true, points3D,
                            diameter=2.0, predict_pose=predict_pose,
                            period=1, verbose=0)
    logs = {}
    callback.on_epoch_end(0, logs)
    assert logs["ADD"] < 1e-6
    assert logs["ADD_accuracy"] == 1.0


def test_evaluate_pose_skips_off_period():
    def predict_pose(model, image):
        return (np.eye(3), np.zeros(3))

    callback = EvaluatePose([np.zeros((4, 4, 3))],
                            [(np.eye(3), np.zeros(3))], unit_points(),
                            diameter=2.0, predict_pose=predict_pose,
                            period=5, verbose=0)
    logs = {}
    callback.on_epoch_end(0, logs)          # epoch 0 -> (0+1)%5 != 0
    assert "ADD" not in logs
