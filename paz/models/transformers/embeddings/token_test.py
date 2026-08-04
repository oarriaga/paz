import numpy as np
from keras import Input, Model

from paz.models.transformers.embeddings.token import LearnableTokens


def test_learnable_tokens_broadcasts_to_batch():
    tokens = Input((5, 8))
    output = LearnableTokens(1, 8, name="cls_token")(tokens)
    model = Model(tokens, output)
    assert tuple(model(np.zeros((3, 5, 8), "float32")).shape) == (3, 1, 8)


def test_learnable_tokens_config_round_trips():
    layer = LearnableTokens(2, 16, name="camera_token")
    restored = LearnableTokens.from_config(layer.get_config())
    assert restored.count == 2 and restored.hidden_size == 16
