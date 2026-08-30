import numpy as np
import jax
from keras import Input, Model, ops

from paz.models.transformers import deformable

GRIDS = ((4, 6), (2, 3))
TOKENS = 4 * 6 + 2 * 3


def build_attention_model(num_heads=2, num_points=3):
    query = Input((5, 8), name="query")
    value = Input((TOKENS, 8), name="value")
    boxes = Input((5, 4), name="boxes")
    args = GRIDS, num_heads, num_points, "cross_attn"
    output = deformable.attend(query, value, boxes, *args)
    return Model([query, value, boxes], output)


def make_inputs(batch=2):
    random = np.random.RandomState(0)
    query = random.randn(batch, 5, 8).astype("float32")
    value = random.randn(batch, TOKENS, 8).astype("float32")
    centers = random.uniform(0.2, 0.8, (batch, 5, 2))
    sizes = random.uniform(0.1, 0.3, (batch, 5, 2))
    boxes = np.concatenate([centers, sizes], axis=-1).astype("float32")
    return query, value, boxes


def test_attention_preserves_query_shape():
    output = build_attention_model()(make_inputs())
    assert tuple(output.shape) == (2, 5, 8)


def test_attention_jit_matches_eager():
    model = build_attention_model()
    inputs = make_inputs()
    eager = np.array(model(inputs))
    jitted = np.array(jax.jit(lambda x: model(x))(list(inputs)))
    assert np.allclose(eager, jitted, atol=1e-5)


def test_single_point_at_box_center_reads_that_pixel():
    values = np.arange(6 * 4, dtype="float32").reshape(1, 6 * 4, 1, 1)
    boxes = np.array([[[0.5 / 6, 1.5 / 4, 0.0, 0.0]]], "float32")
    offsets = np.zeros((1, 1, 1, 1, 1, 2), "float32")
    positions = deformable.compute_positions(boxes, offsets, 1)
    sampled = deformable.sample_levels(values, positions, ((4, 6),))
    assert np.allclose(np.array(sampled)[0, 0, 0, 0], [6.0])


def test_weights_sum_to_one_over_levels_and_points():
    query = Input((5, 8))
    weights = deformable.project_weights(query, 2, 2, 3, "cross_attn")
    model = Model(query, ops.sum(weights, axis=(3, 4)))
    totals = np.array(model(make_inputs()[0]))
    assert np.allclose(totals, 1.0, atol=1e-6)
