import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np

import curriculum


def update(tracking, episode_length, iteration=42):
    args = jp.asarray(0.1), jp.asarray(tracking), jp.asarray(episode_length)
    return float(curriculum.update_max_speed(*args, iteration))


def test_speed_grows_when_tracking_and_surviving():
    assert np.isclose(update(0.95, 1000.0), 0.2)


def test_short_episodes_block_the_curriculum():
    # the reference rates the episodic tracking return against the full
    # episode horizon, so dying early blocks the speed increase
    assert np.isclose(update(0.95, 300.0), 0.1)


def test_poor_tracking_blocks_the_curriculum():
    assert np.isclose(update(0.5, 1000.0), 0.1)


def test_speed_only_changes_on_period_boundaries():
    assert np.isclose(update(0.95, 1000.0, iteration=43), 0.1)


def test_speed_saturates_at_one():
    args = jp.asarray(1.0), jp.asarray(0.95), jp.asarray(1000.0)
    assert np.isclose(float(curriculum.update_max_speed(*args, 42)), 1.0)
