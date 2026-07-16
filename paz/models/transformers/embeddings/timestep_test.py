import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.models.transformers.embeddings import timestep


def test_sinusoidal_matches_closed_form():
    times = np.array([0.0, 0.5, 3.0], "float32")
    embedding = timestep.sinusoidal(times, 8, 100.0, 10.0)
    frequencies = 10.0 * np.exp(-np.log(100.0) * np.arange(4) / 4)
    angles = times[:, None] * frequencies[None]
    expected = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    assert np.allclose(np.asarray(embedding), expected, atol=1e-6)


def test_sinusoidal_at_zero_time():
    times = np.zeros((2,), "float32")
    embedding = np.asarray(timestep.sinusoidal(times, 6, 10.0, 1.0))
    assert np.allclose(embedding[:, :3], 1.0)
    assert np.allclose(embedding[:, 3:], 0.0)
