"""Cached image->text generation for Gemma4.

Images are baked into the KV cache during prefill: each image-placeholder
position is fed its vision embedding (pre-scaled by hidden_dim**-0.5) instead of
a token embedding, matching keras_hub's "interleave before the cached decoder".
Decoding after the prompt is text-only. Uses Gemma4MultimodalDecoderStep, which
takes a precomputed input embedding.
"""
import jax.numpy as jnp
from keras import ops

from .inference import build_empty_cache


def build_prompt_embeddings(step_model, vision_embeddings, prompt_ids,
                            vision_indices, per_layer_step=None):
    embedding = step_model.get_layer("token_embedding")
    image_position = {int(p): k for k, p in enumerate(vision_indices)}
    rows = []
    for index, token in enumerate(prompt_ids):
        if index in image_position:
            row = jnp.asarray(vision_embeddings[image_position[index]])
            embed = row[None, None]
            per_layer = zero_per_layer(per_layer_step, token)
        else:
            embed = embedding(as_token(token))
            per_layer = text_per_layer(per_layer_step, token)
        rows.append((embed, per_layer))
    return rows


def as_token(token):
    return jnp.array([[int(token)]], dtype="int32")


def text_per_layer(per_layer_step, token):
    if per_layer_step is None:
        return None
    return per_layer_step(as_token(token))


def zero_per_layer(per_layer_step, token):
    if per_layer_step is None:
        return None
    return ops.zeros_like(per_layer_step(as_token(token)))


def decode_step(step_model, row, cache, index):
    embed, per_layer = row
    inputs = [embed, cache, jnp.array([index], dtype="int32")]
    if per_layer is not None:
        inputs.append(per_layer)
    return step_model(inputs)


def run_cached_prefill(step_model, config, rows, max_length):
    cache = jnp.asarray(build_empty_cache(config, max_length))
    logits = []
    for index, row in enumerate(rows):
        step_logits, cache = decode_step(step_model, row, cache, index)
        logits.append(step_logits)
    return logits, cache


def generate(step_model, per_layer_step, vision_embeddings, config,
             prompt_ids, vision_indices, stop_id, max_tokens):
    rows = build_prompt_embeddings(
        step_model, vision_embeddings, prompt_ids, vision_indices,
        per_layer_step)
    cache = jnp.asarray(build_empty_cache(config, len(prompt_ids) + max_tokens))
    for index in range(len(rows) - 1):
        _, cache = decode_step(step_model, rows[index], cache, index)
    embedding = step_model.get_layer("token_embedding")
    row, index, generated = rows[-1], len(prompt_ids) - 1, []
    for _ in range(max_tokens):
        logits, cache = decode_step(step_model, row, cache, index)
        token = int(jnp.argmax(logits[0, 0]))
        if token == stop_id:
            break
        generated.append(token)
        index += 1
        per_layer = text_per_layer(per_layer_step, token)
        row = (embedding(as_token(token)), per_layer)
    return generated
