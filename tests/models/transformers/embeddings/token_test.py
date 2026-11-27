import numpy as np
from keras import Input, Model

from paz.models.transformers.embeddings.token import LearnableTokens


def test_learnable_tokens_repeats_its_weight_for_every_batch_item():
    tokens = Input((5, 8))
    output = LearnableTokens(2, 8, name="cls_token")(tokens)
    model = Model(tokens, output)
    weight = np.random.default_rng(0).normal(size=(2, 8)).astype("float32")
    model.set_weights([weight])
    reference = np.ones((3, 5, 8), "float32")
    expected = np.broadcast_to(weight, (3, 2, 8))
    assert np.array_equal(np.array(model(reference)), expected)


def test_learnable_tokens_config_round_trips():
    layer = LearnableTokens(2, 16, name="camera_token")
    restored = LearnableTokens.from_config(layer.get_config())
    assert restored.count == 2 and restored.hidden_size == 16
