"""Constant-velocity Kalman filter over boxes as [cx, cy, W / H, H].

Runs in numpy float64: the initial covariance has a condition number of
about 1e8 * H**2 (1.6e11 at H = 40), more than float32 can resolve.
"""
from collections import namedtuple

import numpy as np
from scipy import linalg

KalmanState = namedtuple("KalmanState", "mean covariance")

POSITION_WEIGHT = 1.0 / 20.0
VELOCITY_WEIGHT = 1.0 / 160.0


def initiate(measurement):
    mean = np.concatenate([measurement, np.zeros(4)])
    args = (mean[3], 2.0 * POSITION_WEIGHT, 10.0 * VELOCITY_WEIGHT)
    return KalmanState(mean, compute_state_noise(*args))


def predict(state):
    args = (state.mean[3], POSITION_WEIGHT, VELOCITY_WEIGHT)
    noise = compute_state_noise(*args)
    motion = np.eye(8) + np.eye(8, k=4)
    covariance = motion @ state.covariance @ motion.T + noise
    return KalmanState(motion @ state.mean, covariance)


def update(state, measurement):
    projected = project(state)
    factor = linalg.cho_factor(projected.covariance, lower=True)
    gain = linalg.cho_solve(factor, state.covariance[:4]).T
    mean = state.mean + gain @ (measurement - projected.mean)
    covariance = state.covariance - gain @ projected.covariance @ gain.T
    return KalmanState(mean, covariance)


def project(state):
    height_stdv = POSITION_WEIGHT * state.mean[3]
    noise = np.diag(compute_variances(height_stdv, 1e-1))
    covariance = state.covariance[:4, :4] + noise
    return KalmanState(state.mean[:4], covariance)


def compute_gating_distances(state, measurements):
    projected = project(state)
    lower = linalg.cholesky(projected.covariance, lower=True)
    residuals = measurements - projected.mean
    whitened = linalg.solve_triangular(lower, residuals.T, lower=True)
    return np.sum(np.square(whitened), axis=0)


def compute_state_noise(height, position_weight, velocity_weight):
    position = compute_variances(position_weight * height, 1e-2)
    velocity = compute_variances(velocity_weight * height, 1e-5)
    return np.diag(np.concatenate([position, velocity]))


def compute_variances(height_stdv, aspect_stdv):
    return np.square([height_stdv, height_stdv, aspect_stdv, height_stdv])


def to_measurement(box2D):
    box2D = np.asarray(box2D, dtype=np.float32)
    center = box2D[:2] + box2D[2:] / 2.0
    aspect_ratio = box2D[2:3] / box2D[3:4]
    return np.concatenate([center, aspect_ratio, box2D[3:4]])


def to_box2D(mean):
    size = np.array([mean[2] * mean[3], mean[3]])
    return np.concatenate([mean[:2] - size / 2.0, size])
