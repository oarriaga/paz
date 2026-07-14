import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax
import jax.numpy as jp

from paz.models.transformers.logits import soft_cap as apply_soft_cap
from paz.models.foundation.gemma4.model import build_text_backbone_args
from paz.models.foundation.gemma4.causal_lm import Gemma4CausalLM
from paz.models.foundation.gemma4.vision import (
    build_vision_encoder_args, num_patches)
from paz.models.foundation.gemma4.multimodal import build_multimodal_backbone
from paz.models.foundation.gemma4.multimodal_decoding import (
    prefill_logits, generate, generate_eager, generate_sample, generate_batch,
    generate_batch_greedy, build_prompt_rows, call_step, trim_to_stop,
    as_token, build_text_generator, build_generator)
from paz.models.foundation.gemma4.sampling import SamplingArgs

TEXT = build_text_backbone_args(num_layers=2, sliding_window_pattern=2)
VISION = build_vision_encoder_args(output_dim=TEXT.hidden_dim)
SCALE = float(TEXT.hidden_dim) ** -0.5


def build_multimodal():
    model = build_multimodal_backbone(TEXT, VISION)
    tokens, grid, pixels, indices = build_inputs()
    model({"token_ids": tokens, "padding_mask": np.ones_like(tokens),
           "pixel_values": pixels, "pixel_position_ids": grid,
           "vision_indices": indices})
    return model


def causal_sharing(multimodal):
    model = Gemma4CausalLM(TEXT)
    model({"token_ids": jp.zeros((1, 1), "int32"),
           "padding_mask": jp.ones((1, 1), "int32")})
    for target, source in zip(model.backbone.weights,
                              multimodal.backbone.weights):
        target.assign(source)
    return model


def full_sequence_logits(multimodal, inputs):
    hidden = multimodal(inputs)
    logits = multimodal.backbone.token_embedding(hidden, reverse=True)
    return apply_soft_cap(logits, TEXT.final_logit_soft_cap)


def image_embeddings(multimodal, pixels, grid):
    images = multimodal.vision_encoder(
        {"pixel_values": pixels, "pixel_position_ids": grid})
    return np.array(images)[0] * SCALE


def build_inputs():
    side = VISION.image_size // VISION.patch_size
    pooled = num_patches(VISION) // VISION.pool_size ** 2
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
    multimodal = build_multimodal()
    model = causal_sharing(multimodal)
    tokens, grid, pixels, indices = build_inputs()
    inputs = {"token_ids": tokens, "padding_mask": np.ones_like(tokens),
              "pixel_values": pixels, "pixel_position_ids": grid,
              "vision_indices": indices}
    full = np.array(full_sequence_logits(multimodal, inputs))[0]
    image = image_embeddings(multimodal, pixels, grid)
    cached = np.array(prefill_logits(model, image, tokens[0], indices[0]))[0]
    assert float(np.max(np.abs(full - cached))) < 1e-4


def reference_decode(model, image, prompt_ids, vision_indices, stop_id,
                     max_tokens):
    embeds, per_layer = build_prompt_rows(
        model, image, prompt_ids, vision_indices)
    length = embeds.shape[1]
    cache = jp.asarray(model.build_cache(length + max_tokens))
    positions = jp.arange(length, dtype="int32")[None]
    logits, cache = call_step(model, embeds, cache, 0, positions, per_layer)
    token = int(jp.argmax(logits[0, -1]))
    out, index = [token], length
    while token != stop_id and len(out) < max_tokens:
        positions = jp.array([[index]], dtype="int32")
        logits, cache = call_step(
            model, model.backbone.token_embedding(as_token(token)), cache,
            index, positions, None)
        token = int(jp.argmax(logits[0, -1]))
        out.append(token)
        index += 1
    return trim_to_stop(out, stop_id)


def test_jitted_decode_matches_reference():
    multimodal = build_multimodal()
    model = causal_sharing(multimodal)
    tokens, grid, pixels, indices = build_inputs()
    image = image_embeddings(multimodal, pixels, grid)
    stop = int(TEXT.vocabulary_size - 1)
    args = (model, image, tokens[0], indices[0], stop, 5)
    got = generate(*args)
    assert len(got) >= 1
    assert got == reference_decode(*args)


def build_text_model():
    return causal_sharing(build_multimodal())


NO_VISION = np.zeros((0, TEXT.hidden_dim), "float32")


def test_generate_batch_greedy_matches_single_sequence():
    model = build_text_model()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    single = generate(model, NO_VISION, prompt, [], stop, 6)
    prompts = np.array([prompt, prompt, prompt], dtype="int32")
    rows = generate_batch_greedy(model, prompts, stop, 6)
    assert rows[0] == single and rows[1] == single and rows[2] == single


def test_generate_eager_matches_jitted_greedy():
    model = build_text_model()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    jitted = generate(model, NO_VISION, prompt, [], stop, 6)
    eager = generate_eager(model, NO_VISION, prompt, [], stop, 6)
    assert eager == jitted


def test_generator_with_vision_matches_eager_reference():
    multimodal = build_multimodal()
    model = causal_sharing(multimodal)
    tokens, grid, pixels, indices = build_inputs()
    image = image_embeddings(multimodal, pixels, grid)
    stop = int(TEXT.vocabulary_size - 1)
    prompt, vision_indices = list(tokens[0]), list(indices[0])
    reference = generate_eager(model, image, prompt, vision_indices, stop, 5)
    decode = build_generator(
        model, stop, max_tokens=5, max_seq=64, max_prompt=32)
    assert decode(prompt, image, vision_indices) == reference


def test_text_generator_matches_reference_greedy():
    model = build_text_model()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    reference = generate(model, NO_VISION, prompt, [], stop, 6)
    decode = build_text_generator(
        model, stop, max_tokens=6, max_seq=64, max_prompt=16)
    assert decode(prompt) == reference


def test_text_generator_streams_tokens_in_order():
    model = build_text_model()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    seen = []
    decode = build_text_generator(
        model, stop, max_tokens=6, max_seq=64, max_prompt=16,
        emit=lambda token_id: seen.append(int(token_id)))
    generated = decode(prompt)
    assert seen[:len(generated)] == generated


def test_generate_sample_peaked_matches_greedy_and_is_seeded():
    model = build_text_model()
    prompt = [2, 5, 9, 11, 7]
    stop = int(TEXT.vocabulary_size - 1)
    greedy = generate(model, NO_VISION, prompt, [], stop, 6)
    peaked = generate_sample(model, NO_VISION, prompt, [], stop, 6,
                             jax.random.PRNGKey(0), SamplingArgs(1.0, 1, 1.0))
    assert peaked == greedy  # top_k=1 == argmax (no float32 ties here)
    args = SamplingArgs(temperature=1.0, top_k=0, top_p=1.0)
    a = generate_sample(model, NO_VISION, prompt, [], stop, 8,
                        jax.random.PRNGKey(5), args)
    b = generate_sample(model, NO_VISION, prompt, [], stop, 8,
                        jax.random.PRNGKey(5), args)
    assert a == b


def test_generate_batch_sampling_is_seeded_and_valid():
    model = build_text_model()
    stop = int(TEXT.vocabulary_size - 1)
    prompts = np.array([[2, 5, 9, 11, 7]] * 4, dtype="int32")
    args = SamplingArgs(temperature=1.0, top_k=0, top_p=1.0)
    rows_a = generate_batch(model, prompts, jax.random.PRNGKey(1), args, stop, 8)
    rows_b = generate_batch(model, prompts, jax.random.PRNGKey(1), args, stop, 8)
    assert rows_a == rows_b  # same key -> reproducible
    flat = [token for row in rows_a for token in row]
    assert all(0 <= token < TEXT.vocabulary_size for token in flat)
    assert len({tuple(row) for row in rows_a}) > 1  # rows diverge
