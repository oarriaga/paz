import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import keras
from keras import Model

from .layers.core import apply_tanh_soft_cap
from .model import build_text_backbone_args
from .vision import build_vision_encoder, build_vision_encoder_args, num_patches
import jax

from .multimodal import build_multimodal_backbone
from .inference import Gemma4MultimodalDecoderStep, build_empty_cache
from .multimodal_decoding import (prefill_logits, generate, generate_eager,
                                  generate_sample, generate_batch,
                                  generate_batch_greedy, build_prompt_rows,
                                  call_step, trim_to_stop, as_token)
from .sampling import SamplingArgs

TEXT = build_text_backbone_args(num_layers=2, sliding_window_pattern=2)
VISION = build_vision_encoder_args(output_dim=TEXT.hidden_dim)


def last2(path):
    path = path.replace("/dense/", "/")  # unwrap clippable einsum
    return "/".join(path.split("/")[-2:])


def transfer(source, target):
    weights = {last2(w.path): np.asarray(keras.ops.convert_to_numpy(w))
               for w in source.weights}
    for variable in target.weights:
        variable.assign(weights[last2(variable.path)].reshape(variable.shape))


def build_full_sequence_causal_lm():
    backbone = build_multimodal_backbone(TEXT, VISION)
    embedding = backbone.get_layer("token_embedding")
    logits = apply_tanh_soft_cap(
        embedding(backbone.output, reverse=True), TEXT.final_logit_soft_cap)
    return Model(backbone.input, logits), backbone


def build_inputs(pooled):
    side = VISION.image_size // VISION.patch_size
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, TEXT.vocabulary_size, (1, 12)).astype("int32")
    cols = np.tile(np.arange(side), side)
    rows = np.repeat(np.arange(side), side)
    grid = np.stack([cols, rows], -1)[None].astype("int32")
    patch_dim = 3 * VISION.patch_size ** 2
    pixels = rng.standard_normal(
        (1, num_patches(VISION), patch_dim)).astype("float32")
    vision_indices = np.arange(2, 2 + pooled)[None].astype("int32")
    return tokens, grid, pixels, vision_indices


def test_cached_multimodal_matches_full_sequence():
    causal, backbone = build_full_sequence_causal_lm()
    step = Gemma4MultimodalDecoderStep(TEXT)
    vision = build_vision_encoder(VISION)
    transfer(backbone, step)
    transfer(backbone, vision)
    pooled = num_patches(VISION) // VISION.pool_size ** 2
    tokens, grid, pixels, vision_indices = build_inputs(pooled)
    full_logits = np.array(causal({
        "token_ids": tokens, "padding_mask": np.ones_like(tokens),
        "pixel_values": pixels, "pixel_position_ids": grid,
        "vision_indices": vision_indices}))[0]
    scale = float(TEXT.hidden_dim) ** -0.5
    image = np.array(vision(
        {"pixel_values": pixels, "pixel_position_ids": grid}))[0] * scale
    cached = np.array(prefill_logits(
        step, None, image, TEXT, tokens[0], vision_indices[0]))[0]
    assert float(np.max(np.abs(full_logits - cached))) < 1e-4


def reference_decode(step, per_layer_step, image, config, prompt_ids,
                     vision_indices, stop_id, max_tokens):
    import jax.numpy as jnp
    embeds, per_layer = build_prompt_rows(
        step, image, prompt_ids, vision_indices, per_layer_step)
    length = embeds.shape[1]
    cache = jnp.asarray(build_empty_cache(config, length + max_tokens))
    positions = jnp.arange(length, dtype="int32")[None]
    logits, cache = call_step(step, embeds, cache, 0, positions, per_layer)
    token = int(jnp.argmax(logits[0, -1]))
    embedding = step.get_layer("token_embedding")
    out, index = [token], length
    while token != stop_id and len(out) < max_tokens:
        positions = jnp.array([[index]], dtype="int32")
        logits, cache = call_step(
            step, embedding(as_token(token)), cache, index, positions, None)
        token = int(jnp.argmax(logits[0, -1]))
        out.append(token)
        index += 1
    return trim_to_stop(out, stop_id)


def test_jitted_decode_matches_reference():
    causal, backbone = build_full_sequence_causal_lm()
    step = Gemma4MultimodalDecoderStep(TEXT)
    vision = build_vision_encoder(VISION)
    transfer(backbone, step)
    transfer(backbone, vision)
    pooled = num_patches(VISION) // VISION.pool_size ** 2
    tokens, grid, pixels, vision_indices = build_inputs(pooled)
    scale = float(TEXT.hidden_dim) ** -0.5
    image = np.array(vision(
        {"pixel_values": pixels, "pixel_position_ids": grid}))[0] * scale
    stop = int(TEXT.vocabulary_size - 1)
    args = (step, None, image, TEXT, tokens[0], vision_indices[0], stop, 5)
    got = generate(*args)
    reference = reference_decode(*args)
    assert len(got) >= 1
    assert got == reference


def build_text_step():
    causal, backbone = build_full_sequence_causal_lm()
    step = Gemma4MultimodalDecoderStep(TEXT)
    transfer(backbone, step)
    return step


def test_generate_batch_greedy_matches_single_sequence():
    step = build_text_step()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    no_vision = np.zeros((0, TEXT.hidden_dim), "float32")
    single = generate(step, None, no_vision, TEXT, prompt, [], stop, 6)
    prompts = np.array([prompt, prompt, prompt], dtype="int32")
    rows = generate_batch_greedy(step, None, TEXT, prompts, stop, 6)
    assert rows[0] == single
    assert rows[1] == single and rows[2] == single


def test_generate_eager_matches_jitted_greedy():
    step = build_text_step()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    nov = np.zeros((0, TEXT.hidden_dim), "float32")
    jitted = generate(step, None, nov, TEXT, prompt, [], stop, 6)
    eager = generate_eager(step, None, nov, TEXT, prompt, [], stop, 6)
    assert eager == jitted


def test_generate_sample_peaked_matches_greedy_and_is_seeded():
    step = build_text_step()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    nov = np.zeros((0, TEXT.hidden_dim), "float32")
    greedy = generate(step, None, nov, TEXT, prompt, [], stop, 6)
    peaked = generate_sample(step, None, nov, TEXT, prompt, [], stop, 6,
                             jax.random.PRNGKey(0), SamplingArgs(1.0, 1, 1.0))
    assert peaked == greedy  # top_k=1 == argmax (no float32 ties here)
    args = SamplingArgs(temperature=1.0, top_k=0, top_p=1.0)
    a = generate_sample(step, None, nov, TEXT, prompt, [], stop, 8,
                        jax.random.PRNGKey(5), args)
    b = generate_sample(step, None, nov, TEXT, prompt, [], stop, 8,
                        jax.random.PRNGKey(5), args)
    assert a == b


def test_generate_batch_sampling_is_seeded_and_valid():
    step = build_text_step()
    stop = int(TEXT.vocabulary_size - 1)
    prompts = np.array([[2, 5, 9, 11, 7]] * 4, dtype="int32")
    args = SamplingArgs(temperature=1.0, top_k=0, top_p=1.0)
    rows_a = generate_batch(step, None, TEXT, prompts,
                            jax.random.PRNGKey(1), args, stop, 8)
    rows_b = generate_batch(step, None, TEXT, prompts,
                            jax.random.PRNGKey(1), args, stop, 8)
    assert rows_a == rows_b  # same key -> reproducible
    flat = [t for row in rows_a for t in row]
    assert all(0 <= t < TEXT.vocabulary_size for t in flat)
    assert len({tuple(r) for r in rows_a}) > 1  # rows diverge under sampling
