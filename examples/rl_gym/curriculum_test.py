import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np

import curriculum


def update(episodic_tracking, iteration=42):
    args = jp.asarray(0.1), jp.asarray(episodic_tracking)
    return float(curriculum.update_max_speed(*args, iteration))


def test_speed_grows_when_the_episodic_tracking_is_earned():
    assert np.isclose(update(0.95), 0.2)


def test_poor_episodic_tracking_blocks_the_curriculum():
    # an episode that tracks well but dies early earns a small return
    assert np.isclose(update(0.5), 0.1)


def test_speed_only_changes_on_period_boundaries():
    assert np.isclose(update(0.95, iteration=43), 0.1)


def test_speed_saturates_at_one():
    args = jp.asarray(1.0), jp.asarray(0.95)
    assert np.isclose(float(curriculum.update_max_speed(*args, 42)), 1.0)
