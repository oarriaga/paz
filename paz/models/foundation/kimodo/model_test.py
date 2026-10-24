import numpy as np
import jax.numpy as jp

from paz.models.foundation.kimodo import features
from paz.models.foundation.kimodo import model as denoiser

TINY = {"hidden_dim": 16, "ff_dim": 24, "num_layers": 2, "num_heads": 4,
        "num_text_tokens": 3, "text_dim": 8}


def build_unit_stats():
    dims = (features.GLOBAL_ROOT_DIM, features.LOCAL_ROOT_DIM,
            features.BODY_DIM)
    arrays = {}
    for block, dim in zip(("global_root", "local_root", "body"), dims):
        arrays[f"{block}_mean"] = np.zeros(dim, "float32")
        arrays[f"{block}_std"] = np.ones(dim, "float32")
    return features.build_stats(arrays)


def build_inputs(num_frames, seed=0):
    rng = np.random.default_rng(seed)
    shape = (2, num_frames, features.FEATURE_DIM)
    noisy = rng.normal(size=shape).astype("float32")
    text = rng.normal(size=(2, 1, TINY["text_dim"])).astype("float32")
    timestep = np.array([3, 500], "int32")
    heading = np.array([0.0, 0.4], "float32")
    zeros = np.zeros(shape, "float32")
    return [noisy, zeros, zeros, text, timestep, heading]


def test_output_shape_and_determinism():
    model = denoiser.build(build_unit_stats(), **TINY)
    inputs = build_inputs(6)
    output = np.asarray(model(inputs))
    assert output.shape == (2, 6, features.FEATURE_DIM)
    assert np.allclose(output, np.asarray(model(inputs)))
    assert np.isfinite(output).all()


def test_frame_count_is_dynamic():
    model = denoiser.build(build_unit_stats(), **TINY)
    assert model(build_inputs(6)).shape[1] == 6
    assert model(build_inputs(9)).shape[1] == 9


def test_masked_values_ignore_the_noisy_input():
    model = denoiser.build(build_unit_stats(), **TINY)
    inputs = build_inputs(5)
    mask = np.ones_like(inputs[1])
    observed = np.asarray(inputs[0])
    masked = [observed * 0.0 + 7.0, mask, observed, *inputs[3:]]
    reference = np.asarray(model([observed, mask, observed, *inputs[3:]]))
    assert np.allclose(np.asarray(model(masked)), reference, atol=1e-5)


def test_position_embedding_interleaves_sine_and_cosine():
    embedding = np.asarray(denoiser.embed_position(jp.array([1.0]), 4, (4,)))
    frequency = 10000.0 ** (-2.0 / 4)
    expected = [np.sin(1.0), np.cos(1.0), np.sin(frequency), np.cos(frequency)]
    assert np.allclose(embedding[0], expected, atol=1e-6)


def test_root_to_local_constant_velocity():
    stats = build_unit_stats()
    fps = 30.0
    frames = np.arange(4, dtype="float32")
    position = np.stack([frames * 0.1, np.full(4, 0.8), -frames * 0.2], -1)
    heading = frames * 0.05
    root = np.concatenate(
        [position, np.cos(heading)[:, None], np.sin(heading)[:, None]], -1)
    local = np.asarray(denoiser.convert_root_to_local(root[None], stats, fps))
    expected_rate = np.full(4, 0.05 * fps)
    assert np.allclose(local[0, :, 0], expected_rate, atol=1e-4)
    assert np.allclose(local[0, :, 1], 0.1 * fps, atol=1e-4)
    assert np.allclose(local[0, :, 2], -0.2 * fps, atol=1e-4)
    assert np.allclose(local[0, :, 3], 0.8, atol=1e-6)
