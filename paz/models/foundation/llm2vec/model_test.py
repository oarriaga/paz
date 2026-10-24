import jax.numpy as jp
import numpy as np

from paz.models.foundation.llm2vec import model as encoder

TINY = {"vocab_size": 50, "hidden_dim": 16, "num_layers": 2, "num_heads": 4,
        "num_kv_heads": 2, "head_dim": 4, "ff_dim": 24, "dtype": "float32"}


def test_hidden_state_shape_and_dtype():
    model = encoder.build(**TINY)
    hidden = model(np.array([[3, 7, 11, 2, 9]], "int32"))
    assert hidden.shape == (1, 5, 16)
    assert np.isfinite(np.asarray(hidden)).all()


def test_attention_is_bidirectional():
    model = encoder.build(**TINY)
    base = np.asarray(model(np.array([[3, 7, 11, 2, 9]], "int32")))
    changed = np.asarray(model(np.array([[3, 7, 11, 2, 1]], "int32")))
    assert not np.allclose(base[0, 0], changed[0, 0])


def test_grouped_attention_matches_per_head_attention():
    rng = np.random.default_rng(0)
    query = rng.normal(size=(1, 5, 4, 3)).astype("float32")
    key = rng.normal(size=(1, 5, 2, 3)).astype("float32")
    value = rng.normal(size=(1, 5, 2, 3)).astype("float32")
    context = np.asarray(encoder.attend_groups(jp.asarray(query),
                                               jp.asarray(key),
                                               jp.asarray(value), 2))
    for head in range(4):
        kv_head = head // 2
        scores = query[0, :, head] @ key[0, :, kv_head].T / np.sqrt(3.0)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights = weights / weights.sum(-1, keepdims=True)
        expected = weights @ value[0, :, kv_head]
        assert np.allclose(context[0, :, head], expected, atol=1e-5)


def test_bfloat16_weights_by_default():
    model = encoder.build(**dict(TINY, dtype="bfloat16"))
    kernel = model.get_layer("layer_0_attention_query").kernel
    assert kernel.dtype == "bfloat16"
