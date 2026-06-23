import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import keras
from keras import Model

from .layers.core import apply_tanh_soft_cap
from .model import build_text_backbone_args
from .vision import build_vision_encoder, build_vision_encoder_args, num_patches
from .multimodal import build_multimodal_backbone
from .inference import Gemma4MultimodalDecoderStep
from .multimodal_decoding import build_prompt_embeddings, run_cached_prefill

TEXT = build_text_backbone_args(num_layers=2, sliding_window_pattern=2)
VISION = build_vision_encoder_args(output_dim=TEXT.hidden_dim)


def last2(path):
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
    rows = build_prompt_embeddings(step, image, tokens[0], vision_indices[0])
    cached, _ = run_cached_prefill(step, TEXT, rows, tokens.shape[1])
    cached_logits = np.stack([np.array(c)[0, 0] for c in cached])
    assert float(np.max(np.abs(full_logits - cached_logits))) < 1e-4
