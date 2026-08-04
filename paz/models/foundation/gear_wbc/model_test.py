import numpy as np

from paz.models.foundation.gear_wbc.model import ACTION_DIM
from paz.models.foundation.gear_wbc.model import LATENT_DIM
from paz.models.foundation.gear_wbc.model import OBSERVATION_DIM
from paz.models.foundation.gear_wbc.model import build_actor
from paz.models.foundation.gear_wbc.model import compute_unit_norm


def test_actor_maps_observation_history_to_lower_body_action():
    actor = build_actor()
    x = np.zeros((1, OBSERVATION_DIM), dtype="float32")
    action = np.array(actor(x, training=False))
    assert action.shape == (1, ACTION_DIM)


def test_actor_parameter_count_matches_release():
    assert build_actor().count_params() == 470578


def test_older_frames_reach_the_action_through_the_estimator():
    # The actor trunk is fed the last frame only, so an older frame can move
    # the action only through the estimator. This pins that path as wired.
    actor = build_actor()
    x = np.zeros((1, OBSERVATION_DIM), dtype="float32")
    older = x.copy()
    older[0, 0] = 1.0
    assert not np.allclose(actor(x, training=False),
                           actor(older, training=False))


def test_unit_norm_rescales_latent_to_unit_length():
    latent = np.random.default_rng(0).normal(size=(4, LATENT_DIM))
    norms = np.linalg.norm(np.array(compute_unit_norm(latent)), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_unit_norm_keeps_zero_latent_finite():
    latent = np.zeros((1, LATENT_DIM), dtype="float32")
    assert np.isfinite(np.array(compute_unit_norm(latent))).all()
