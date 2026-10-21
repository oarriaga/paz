import numpy as np

from paz.models.tracking.deep_sort.kalman import KalmanState
from paz.models.tracking.deep_sort.kalman import initiate
from paz.models.tracking.deep_sort.kalman import predict
from paz.models.tracking.deep_sort.kalman import project
from paz.models.tracking.deep_sort.kalman import update
from paz.models.tracking.deep_sort.kalman import compute_gating_distances
from paz.models.tracking.deep_sort.kalman import to_measurement
from paz.models.tracking.deep_sort.kalman import to_box2D

MEASUREMENT = np.array([100.0, 50.0, 0.5, 40.0])


def build_moving_state():
    mean = np.concatenate([MEASUREMENT, [3.0, -2.0, 0.01, 10.0]])
    stdv = np.array([4.0, 4.0, 1e-2, 4.0, 2.5, 2.5, 1e-5, 2.5])
    covariance = np.diag(np.square(stdv))
    return KalmanState(mean, covariance)


def build_motion_matrix():
    return np.eye(8) + np.eye(8, k=4)


def compute_closed_form_update(state, measurement):
    observation = np.eye(4, 8)
    height = state.mean[3]
    stdv = np.array([0.05 * height, 0.05 * height, 1e-1, 0.05 * height])
    projected = observation @ state.covariance @ observation.T
    innovation_covariance = projected + np.diag(np.square(stdv))
    inverse = np.linalg.inv(innovation_covariance)
    gain = state.covariance @ observation.T @ inverse
    residual = measurement - observation @ state.mean
    mean = state.mean + gain @ residual
    covariance = state.covariance - gain @ innovation_covariance @ gain.T
    return mean, covariance


def test_initiate_copies_measurement_with_zero_velocity():
    state = initiate(MEASUREMENT)
    expected = [100.0, 50.0, 0.5, 40.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(state.mean, expected, rtol=0, atol=1e-12)


def test_initiate_covariance_diagonal_for_height_40():
    covariance = initiate(MEASUREMENT).covariance
    diagonal = [16.0, 16.0, 1e-4, 16.0, 6.25, 6.25, 1e-10, 6.25]
    assert np.allclose(covariance, np.diag(diagonal), rtol=1e-9, atol=0)


def test_initiate_covariance_is_float64():
    measurement = MEASUREMENT.astype(np.float32)
    assert initiate(measurement).covariance.dtype == np.float64


def test_initiate_covariance_is_too_ill_conditioned_for_float32():
    condition = np.linalg.cond(initiate(MEASUREMENT).covariance)
    assert np.isclose(condition, 1.6e11, rtol=1e-6)
    assert condition > 1.0 / np.finfo(np.float32).eps


def test_predict_moves_mean_by_one_step_of_velocity():
    mean = predict(build_moving_state()).mean
    expected = [103.0, 48.0, 0.51, 50.0, 3.0, -2.0, 0.01, 10.0]
    assert np.allclose(mean, expected, rtol=0, atol=1e-12)


def test_predict_noise_uses_height_before_prediction():
    covariance = predict(build_moving_state()).covariance
    assert np.isclose(covariance[0, 0], 16.0 + 6.25 + 4.0, rtol=1e-12)
    assert np.isclose(covariance[4, 4], 6.25 + 0.0625, rtol=1e-12)
    assert np.isclose(covariance[0, 4], 6.25, rtol=1e-12)


def test_predict_aspect_noise():
    covariance = predict(build_moving_state()).covariance
    assert np.isclose(covariance[2, 2], 2e-4 + 1e-10, rtol=1e-9, atol=0)
    assert np.isclose(covariance[6, 6], 2e-10, rtol=1e-9, atol=0)


def test_predict_covariance_matches_closed_form():
    state = build_moving_state()
    motion = build_motion_matrix()
    stdv = np.array([2.0, 2.0, 1e-2, 2.0, 0.25, 0.25, 1e-5, 0.25])
    noise = np.diag(np.square(stdv))
    expected = motion @ state.covariance @ motion.T + noise
    covariance = predict(state).covariance
    assert np.allclose(covariance, expected, rtol=1e-12, atol=1e-15)


def test_project_adds_measurement_noise():
    state = project(initiate(MEASUREMENT))
    expected = np.diag([20.0, 20.0, 1.01e-2, 20.0])
    assert np.allclose(state.mean, MEASUREMENT, rtol=0, atol=1e-12)
    assert np.allclose(state.covariance, expected, rtol=1e-9, atol=0)


def test_update_matches_closed_form():
    state = predict(build_moving_state())
    measurement = np.array([104.0, 47.5, 0.52, 49.0])
    mean, covariance = compute_closed_form_update(state, measurement)
    updated = update(state, measurement)
    assert np.allclose(updated.mean, mean, rtol=0, atol=1e-9)
    assert np.allclose(updated.covariance, covariance, rtol=0, atol=1e-9)


def test_update_with_exact_measurement_keeps_zero_velocity():
    updated = update(initiate(MEASUREMENT), MEASUREMENT)
    expected = [100.0, 50.0, 0.5, 40.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(updated.mean, expected, rtol=0, atol=1e-12)


def test_gating_distances_for_single_dimension_offsets():
    measurements = MEASUREMENT + np.diag([2.0, 0.0, 0.1, 2.0])
    distances = compute_gating_distances(initiate(MEASUREMENT), measurements)
    expected = [0.2, 0.0, 1.0 / 1.01, 0.2]
    assert np.allclose(distances, expected, rtol=1e-9, atol=1e-12)


def test_gating_distances_match_closed_form_mahalanobis():
    state = predict(build_moving_state())
    near = [104.0, 47.5, 0.52, 49.0]
    far = [90.0, 60.0, 0.4, 55.0]
    measurements = np.array([near, far])
    projected = project(state)
    inverse = np.linalg.inv(projected.covariance)
    residuals = measurements - projected.mean
    expected = np.sum((residuals @ inverse) * residuals, axis=1)
    distances = compute_gating_distances(state, measurements)
    assert np.allclose(distances, expected, rtol=1e-9, atol=1e-9)


def test_to_measurement_converts_left_top_width_height():
    box2D = np.array([10.0, 10.0, 20.0, 40.0], np.float32)
    expected = [20.0, 30.0, 0.5, 40.0]
    assert np.allclose(to_measurement(box2D), expected, rtol=0, atol=1e-6)


def test_to_measurement_keeps_float32():
    box2D = [10.0, 10.0, 20.0, 40.0]
    assert to_measurement(box2D).dtype == np.float32
    assert to_measurement(np.array(box2D, np.float32)).dtype == np.float32


def test_to_box2D_inverts_measurement():
    mean = np.array([20.0, 30.0, 0.5, 40.0, 1.0, 2.0, 0.0, 3.0])
    expected = [10.0, 10.0, 20.0, 40.0]
    assert np.allclose(to_box2D(mean), expected, rtol=0, atol=1e-12)
