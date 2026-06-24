"""Cached image->text generation for Gemma4.

Image embeddings are baked into the KV cache during prefill: each image
placeholder position is fed its vision embedding (pre-scaled by
hidden_dim**-0.5) instead of a token embedding, matching keras_hub. The whole
prompt is prefilled in ONE forward (parallel prefill); decode is then a jitted
jax.lax.while_loop, mirroring decoding.py's text KVDecoder. Uses
Gemma4MultimodalDecoderStep, which takes a precomputed input embedding and the
query positions.
"""
import jax
import jax.numpy as jnp
import numpy as np
from keras import ops

from .inference import build_empty_cache
from .sampling import sample_logits


def as_token(token):
    return jnp.array([[int(token)]], dtype="int32")


def build_prompt_rows(step_model, vision_embeddings, prompt_ids,
                      vision_indices, per_layer_step):
    embedding = step_model.get_layer("token_embedding")
    image_position = {int(p): k for k, p in enumerate(vision_indices)}
    embeds, per_layers = [], []
    for index, token in enumerate(prompt_ids):
        if index in image_position:
            embeds.append(jnp.asarray(vision_embeddings[image_position[index]]))
            per_layers.append(per_layer_row(per_layer_step, 0))
        else:
            embeds.append(embedding(as_token(token))[0, 0])
            per_layers.append(per_layer_row(per_layer_step, token))
    input_embeddings = jnp.stack(embeds)[None]
    per_layer = None
    if per_layer_step is not None:
        per_layer = jnp.stack(per_layers)[None]
    return input_embeddings, per_layer


def per_layer_row(per_layer_step, token):
    # Image positions reuse the pad-token (id 0) per-layer embedding, matching
    # keras_hub which zeroes the per-layer token ids at vision positions.
    if per_layer_step is None:
        return None
    return per_layer_step(as_token(token))[0, 0]


def call_step(step_model, embeds, cache, start, positions, per_layer):
    index = jnp.reshape(jnp.asarray(start, dtype="int32"), (1,))
    inputs = [embeds, cache, index, positions]
    if per_layer is not None:
        inputs.append(per_layer)
    return step_model(inputs)


def generate(step_model, per_layer_step, vision_embeddings, config,
             prompt_ids, vision_indices, stop_id, max_tokens):
    embeds, per_layer = build_prompt_rows(
        step_model, vision_embeddings, prompt_ids, vision_indices,
        per_layer_step)
    length = embeds.shape[1]
    max_length = length + max_tokens
    cache = jnp.asarray(build_empty_cache(config, max_length))
    positions = jnp.arange(length, dtype="int32")[None]
    logits, cache = call_step(
        step_model, embeds, cache, 0, positions, per_layer)
    first = jnp.argmax(logits[0, -1]).astype("int32")
    decode = build_decode_loop(step_model, per_layer_step, max_length,
                               max_tokens)
    buffer, count = decode(cache, first, jnp.array(length, "int32"),
                           jnp.array(stop_id, "int32"))
    generated = ops.convert_to_numpy(buffer[:int(count)]).tolist()
    return trim_to_stop(generated, stop_id)


def trim_to_stop(tokens, stop_id):
    if stop_id in tokens:
        tokens = tokens[:tokens.index(stop_id)]
    return tokens


def build_decode_loop(step_model, per_layer_step, max_length, max_tokens):
    embedding = step_model.get_layer("token_embedding")

    @jax.jit
    def decode(cache, first_token, start_index, stop_id):
        buffer = jnp.zeros((max_tokens,), dtype="int32").at[0].set(first_token)
        count = jnp.array(1, dtype="int32")
        state = (buffer, first_token, start_index, cache, count,
                 first_token == stop_id)
        cont = build_should_continue(max_tokens, max_length)
        step = build_decode_step(step_model, embedding, per_layer_step, stop_id)
        buffer, _, _, _, count, _ = jax.lax.while_loop(cont, step, state)
        return buffer, count

    return decode


def build_should_continue(max_tokens, max_length):
    def check(state):
        _, _, index, _, count, finished = state
        return (~finished) & (count < max_tokens) & (index < max_length)
    return check


def build_decode_step(step_model, embedding, per_layer_step, stop_id):
    def step(state):
        buffer, token, index, cache, count, _ = state
        token2d = jnp.reshape(token, (1, 1))
        positions = jnp.reshape(index, (1, 1)).astype("int32")
        per_layer = None
        if per_layer_step is not None:
            per_layer = per_layer_step(token2d)
        logits, cache = call_step(
            step_model, embedding(token2d), cache, index, positions, per_layer)
        next_id = jnp.argmax(logits[:, 0, :], axis=-1).astype("int32")[0]
        buffer = buffer.at[count].set(next_id)
        finished = next_id == stop_id
        return (buffer, next_id, index + 1, cache, count + 1, finished)
    return step


def generate_batch(step_model, per_layer_step, config, prompts, key, sampling,
                   stop_id, max_tokens):
    """Batched sampling generation for equal-length text prompts.

    Each row is sampled independently with `sampling`; `key` seeds the draws.
    """
    select = build_sampler(sampling)
    return run_batch_decode(step_model, per_layer_step, config, prompts, key,
                            select, stop_id, max_tokens)


def generate_batch_greedy(step_model, per_layer_step, config, prompts,
                          stop_id, max_tokens):
    """Batched greedy (argmax) generation for equal-length text prompts."""
    return run_batch_decode(step_model, per_layer_step, config, prompts, None,
                            select_greedy, stop_id, max_tokens)


def build_sampler(sampling):
    return lambda logits, key: sample_logits(logits, key, sampling)


def select_greedy(logits, key):
    return ops.cast(ops.argmax(logits, axis=-1), "int32")


def pick_token(select, logits, key):
    if key is None:
        return None, select(logits, None)
    key, step_key = jax.random.split(key)
    return key, select(logits, step_key)


def run_batch_decode(step_model, per_layer_step, config, prompts, key, select,
                     stop_id, max_tokens):
    """Eager parallel prefill + lockstep decode over a (batch, length) grid.

    All rows share query positions, so the cached attention broadcasts across
    the batch. Returns one stop-trimmed token list per row.
    """
    prompts = jnp.asarray(prompts, dtype="int32")
    batch, length = prompts.shape
    embeds, per_layer = build_text_batch_rows(
        step_model, per_layer_step, prompts)
    cache = jnp.asarray(build_empty_cache(config, length + max_tokens, batch))
    positions = jnp.arange(length, dtype="int32")[None]
    logits, cache = call_step(
        step_model, embeds, cache, 0, positions, per_layer)
    key, token = pick_token(select, logits[:, -1], key)
    embedding = step_model.get_layer("token_embedding")
    collected, index = [token], length
    finished = np.asarray(token) == stop_id
    while len(collected) < max_tokens and not finished.all():
        token2d = token[:, None]
        per_layer = batch_per_layer(per_layer_step, token2d)
        logits, cache = call_step(
            step_model, embedding(token2d), cache, index,
            jnp.full((1, 1), index, dtype="int32"), per_layer)
        key, token = pick_token(select, logits[:, -1], key)
        collected.append(token)
        finished = finished | (np.asarray(token) == stop_id)
        index += 1
    return trim_rows(collected, stop_id)


def build_text_batch_rows(step_model, per_layer_step, prompts):
    embedding = step_model.get_layer("token_embedding")
    embeds = embedding(prompts)
    per_layer = batch_per_layer(per_layer_step, prompts)
    return embeds, per_layer


def batch_per_layer(per_layer_step, tokens):
    if per_layer_step is None:
        return None
    return per_layer_step(tokens)


def trim_rows(collected, stop_id):
    grid = np.stack([np.asarray(token) for token in collected], axis=1)
    return [trim_to_stop(row.tolist(), stop_id) for row in grid]


def prefill_logits(step_model, per_layer_step, vision_embeddings, config,
                   prompt_ids, vision_indices):
    """Prefill the whole prompt in one forward; returns per-position logits."""
    embeds, per_layer = build_prompt_rows(
        step_model, vision_embeddings, prompt_ids, vision_indices,
        per_layer_step)
    length = embeds.shape[1]
    cache = jnp.asarray(build_empty_cache(config, length))
    positions = jnp.arange(length, dtype="int32")[None]
    logits, _ = call_step(step_model, embeds, cache, 0, positions, per_layer)
    return logits
