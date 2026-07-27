import numpy as np

from paz.models.foundation.gear_wbc.model import ACTION_DIM
from paz.models.foundation.gear_wbc.model import OBSERVATION_DIM
from paz.models.foundation.gear_wbc.pretrained import GearWBC


def test_pretrained_gear_wbc_downloads_and_runs():
    models = GearWBC(weights="pretrained")
    x = np.zeros((1, OBSERVATION_DIM), dtype="float32")
    for actor in (models.balance, models.walk):
        action = np.array(actor(x, training=False))
        assert action.shape == (1, ACTION_DIM)
        assert np.isfinite(action).all()
