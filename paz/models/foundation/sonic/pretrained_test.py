import numpy as np

from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.pretrained import SONIC


def test_pretrained_sonic_downloads_and_runs():
    sonic = SONIC(weights="pretrained")
    layout = sonic.layout
    x = np.zeros((1, compute_encoder_input_dim(layout)), dtype="float32")
    tokens = np.array(sonic.encoder(x, training=False))
    assert tokens.shape == (1, layout.token_dim)
    assert np.isfinite(tokens).all()
