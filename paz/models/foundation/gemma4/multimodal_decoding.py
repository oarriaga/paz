"""Cached image->text (and batched text) generation for Gemma4.

Image embeddings are baked into the KV cache during prefill: each image
placeholder position is fed its vision embedding (pre-scaled by
hidden_dim**-0.5) instead of a token embedding, matching keras_hub. The whole
prompt is prefilled in ONE forward (parallel prefill); decode is then a jitted
jax.lax.while_loop. All paths run over one Gemma4CausalLM through its
`call_with_cache`; the jitted loop threads the model weights as explicit
arguments via keras.StatelessScope so jax.jit does not constant-fold the
multi-GB embedding tables into the executable.
"""
import itertools

import jax
import jax.numpy as jp
import numpy as np
import keras
from keras import ops

from paz import snapshot_variables

from paz.models.transformers.search import discard, emit_token
from paz.models.foundation.gemma4.sampling import sample_logits


def as_token(token):
    return jp.array([[int(token)]], dtype="int32")


def build_prompt_rows(model, vision_embeddings, prompt_ids, vision_indices):
    embedding = model.backbone.token_embedding
    ids = jp.array(prompt_ids, dtype="int32")[None]
    embeds = embedding(ids)[0]
    per_layer = model.backbone.per_layer_lookup(ids)
    zero_row = per_layer_pad_row(model)
    for image_index, position in enumerate(vision_indices):
        row = jp.asarray(vision_embeddings[image_index])
        embeds = embeds.at[int(position)].set(row)
        if per_layer is not None:
            per_layer = per_layer.at[0, int(position)].set(zero_row)
    return embeds[None], per_layer


def per_layer_pad_row(model):
    # Image positions reuse the pad-token (id 0) per-layer embedding, matching
    # keras_hub which zeroes the per-layer token ids at vision positions.
    if not model.backbone.has_per_layer:
        return None
    return model.backbone.per_layer_lookup(as_token(0))[0, 0]


def call_step(model, embeds, cache, start, positions, per_layer):
    index = jp.asarray(start, dtype="int32")
    return model.call_with_cache(embeds, cache, index, positions, per_layer)


def generate(model, vision_embeddings, prompt_ids, vision_indices, stop_id,
             max_tokens):
    """Single-sequence jitted greedy generation."""
    return run_cached_generate(
        model, vision_embeddings, prompt_ids, vision_indices, stop_id,
        max_tokens, jax.random.PRNGKey(0), select_greedy_token)


def generate_sample(model, vision_embeddings, prompt_ids, vision_indices,
                    stop_id, max_tokens, key, sampling):
    """Single-sequence jitted sampling generation seeded by `key`."""
    return run_cached_generate(
        model, vision_embeddings, prompt_ids, vision_indices, stop_id,
        max_tokens, key, build_token_sampler(sampling))


def generate_eager(model, vision_embeddings, prompt_ids, vision_indices,
                   stop_id, max_tokens):
    """Eager parallel prefill + Python greedy decode loop.

    Prefills the whole prompt in one forward, then decodes token-by-token. Used
    for the largest weights, where a full-program jit would constant-fold the
    embedding tables and exhaust the host.
    """
    embeds, per_layer = build_prompt_rows(
        model, vision_embeddings, prompt_ids, vision_indices)
    length = embeds.shape[1]
    cache = jp.asarray(model.build_cache(length + max_tokens))
    positions = jp.arange(length, dtype="int32")[None]
    logits, cache = call_step(model, embeds, cache, 0, positions, per_layer)
    token = int(jp.argmax(logits[0, -1]))
    out, index = [token], length
    while token != stop_id and len(out) < max_tokens:
        embeds = model.backbone.token_embedding(as_token(token))
        per_layer = decode_per_layer(model, token)
        positions = jp.array([[index]], dtype="int32")
        logits, cache = call_step(
            model, embeds, cache, index, positions, per_layer)
        token = int(jp.argmax(logits[0, -1]))
        out.append(token)
        index += 1
    return trim_to_stop(out, stop_id)


def decode_per_layer(model, token):
    if not model.backbone.has_per_layer:
        return None
    return model.backbone.per_layer_lookup(as_token(token))


def select_greedy_token(logits, key):
    return jp.argmax(logits[0]).astype("int32")


def build_token_sampler(sampling):
    return lambda logits, key: sample_logits(logits, key, sampling)[0]


def run_cached_generate(model, vision_embeddings, prompt_ids, vision_indices,
                        stop_id, max_tokens, key, select):
    embeds, per_layer = build_prompt_rows(
        model, vision_embeddings, prompt_ids, vision_indices)
    length = embeds.shape[1]
    max_length = length + max_tokens
    cache = jp.asarray(model.build_cache(max_length))
    positions = jp.arange(length, dtype="int32")[None]
    logits, cache = call_step(model, embeds, cache, 0, positions, per_layer)
    key, first_key = jax.random.split(key)
    first = select(logits[:, -1], first_key)
    decode = build_decode_loop(model, max_length, max_tokens, select)
    variables = snapshot_variables(model)
    buffer, count = decode(cache, first, jp.array(length, "int32"),
                           jp.array(stop_id, "int32"), key, variables)
    generated = ops.convert_to_numpy(buffer[:int(count)]).tolist()
    return trim_to_stop(generated, stop_id)


def trim_to_stop(tokens, stop_id):
    if stop_id in tokens:
        tokens = tokens[:tokens.index(stop_id)]
    return tokens


def build_generator(model, stop_id, max_tokens, max_seq, max_prompt,
                    select=select_greedy_token, emit=discard):
    """Compile-once greedy generator for text and image prompts alike.

    Returns `generate(prompt_ids, vision_embeddings, vision_indices)`. Every
    prompt is right-padded to `max_prompt` so prefill keeps one static shape and
    the JIT compiles once; the decode loop uses a fixed `max_seq` cache. Causal
    attention keeps padded positions out of the real tokens, so starting decode
    at the true prompt length is exact. `emit(token_id)` fires per token for
    streaming; pass `discard` to stay silent.
    """
    assert max_prompt + max_tokens <= max_seq
    positions = jp.arange(max_prompt, dtype="int32")[None]
    decode = build_streaming_decode_loop(
        model, max_seq, max_tokens, select, emit)
    variables = snapshot_variables(model)

    def generate(prompt_ids, vision_embeddings, vision_indices):
        length = min(len(prompt_ids), max_prompt)
        padded = pad_prompt_ids(prompt_ids, max_prompt)
        embeds, per_layer = build_prompt_rows(
            model, vision_embeddings, padded, vision_indices)
        cache = jp.asarray(model.build_cache(max_seq))
        logits, cache = call_step(model, embeds, cache, 0, positions, per_layer)
        first = select(logits[:, length - 1], None)
        emit(first)
        buffer, count = decode(
            cache, first, jp.array(length, "int32"),
            jp.array(stop_id, "int32"), jax.random.PRNGKey(0), variables)
        generated = ops.convert_to_numpy(buffer[:int(count)]).tolist()
        return trim_to_stop(generated, stop_id)

    return generate


def build_text_generator(model, stop_id, max_tokens, max_seq, max_prompt,
                         select=select_greedy_token, emit=discard):
    """Text specialization of build_generator: `generate(prompt_ids)`."""
    args = (model, stop_id, max_tokens, max_seq, max_prompt, select, emit)
    generate = build_generator(*args)
    no_vision = np.zeros((0, model.config.hidden_dim), "float32")

    def generate_text(prompt_ids):
        return generate(prompt_ids, no_vision, [])

    return generate_text


def pad_prompt_ids(prompt_ids, max_prompt):
    prompt_ids = list(prompt_ids[:max_prompt])
    return prompt_ids + [0] * (max_prompt - len(prompt_ids))


def build_decode_loop(model, max_length, max_tokens, select):
    return build_streaming_decode_loop(
        model, max_length, max_tokens, select, discard)


def build_streaming_decode_loop(model, max_length, max_tokens, select, emit):
    """Jitted decode loop calling `emit(token_id)` per decoded token.

    `emit` runs on the host from inside the jitted loop via jax.debug.callback,
    so tokens can be shown as they are produced. Pass `discard` for no output.
    """
    @jax.jit
    def decode(cache, first_token, start_index, stop_id, key, variables):
        buffer = jp.zeros((max_tokens,), dtype="int32").at[0].set(first_token)
        count = jp.array(1, dtype="int32")
        state = (buffer, first_token, start_index, cache, count,
                 first_token == stop_id, key)
        cont = build_should_continue(max_tokens, max_length)
        step = build_decode_step(model, stop_id, select, variables, emit)
        buffer, _, _, _, count, _, _ = jax.lax.while_loop(cont, step, state)
        return buffer, count

    return decode


def build_should_continue(max_tokens, max_length):
    def check(state):
        _, _, index, _, count, finished, _ = state
        return (~finished) & (count < max_tokens) & (index < max_length)
    return check


def build_decode_step(model, stop_id, select, variables, emit):
    def step(state):
        buffer, token, index, cache, count, _, key = state
        positions = jp.reshape(index, (1, 1)).astype("int32")
        logits, cache = cached_step_stateless(
            model, variables, token, cache, index, positions)
        key, step_key = jax.random.split(key)
        next_id = select(logits[:, 0, :], step_key)
        emit_token(emit, next_id)
        buffer = buffer.at[count].set(next_id)
        finished = next_id == stop_id
        return (buffer, next_id, index + 1, cache, count + 1, finished, key)
    return step


def cached_step_stateless(model, variables, token, cache, index, positions):
    mapping = itertools.chain(
        zip(model.trainable_variables, variables[0]),
        zip(model.non_trainable_variables, variables[1]))
    token2d = jp.reshape(token, (1, 1))
    with keras.StatelessScope(state_mapping=mapping):
        embeds = model.backbone.token_embedding(token2d)
        per_layer = model.backbone.per_layer_lookup(token2d)
        return model.call_with_cache(embeds, cache, index, positions, per_layer)


def generate_batch(model, prompts, key, sampling, stop_id, max_tokens):
    """Batched sampling generation for equal-length text prompts."""
    select = build_sampler(sampling)
    return run_batch_decode(
        model, prompts, key, select, stop_id, max_tokens)


def generate_batch_greedy(model, prompts, stop_id, max_tokens):
    """Batched greedy (argmax) generation for equal-length text prompts."""
    return run_batch_decode(
        model, prompts, None, select_greedy, stop_id, max_tokens)


def build_sampler(sampling):
    return lambda logits, key: sample_logits(logits, key, sampling)


def select_greedy(logits, key):
    return ops.cast(ops.argmax(logits, axis=-1), "int32")


def pick_token(select, logits, key):
    if key is None:
        return None, select(logits, None)
    key, step_key = jax.random.split(key)
    return key, select(logits, step_key)


def run_batch_decode(model, prompts, key, select, stop_id, max_tokens):
    """Eager parallel prefill + lockstep decode over a (batch, length) grid."""
    prompts = jp.asarray(prompts, dtype="int32")
    batch, length = prompts.shape
    embeds = model.backbone.token_embedding(prompts)
    per_layer = batch_per_layer(model, prompts)
    cache = jp.asarray(model.build_cache(length + max_tokens, batch))
    positions = jp.arange(length, dtype="int32")[None]
    logits, cache = call_step(model, embeds, cache, 0, positions, per_layer)
    key, token = pick_token(select, logits[:, -1], key)
    collected, index = [token], length
    finished = np.asarray(token) == stop_id
    while len(collected) < max_tokens and not finished.all():
        token2d = token[:, None]
        per_layer = batch_per_layer(model, token2d)
        positions = jp.full((1, 1), index, dtype="int32")
        logits, cache = call_step(
            model, model.backbone.token_embedding(token2d), cache, index,
            positions, per_layer)
        key, token = pick_token(select, logits[:, -1], key)
        collected.append(token)
        finished = finished | (np.asarray(token) == stop_id)
        index += 1
    return trim_rows(collected, stop_id)


def batch_per_layer(model, tokens):
    if not model.backbone.has_per_layer:
        return None
    return model.backbone.per_layer_lookup(tokens)


def trim_rows(collected, stop_id):
    grid = np.stack([np.asarray(token) for token in collected], axis=1)
    return [trim_to_stop(row.tolist(), stop_id) for row in grid]


def prefill_logits(model, vision_embeddings, prompt_ids, vision_indices):
    """Prefill the whole prompt in one forward; returns per-position logits."""
    embeds, per_layer = build_prompt_rows(
        model, vision_embeddings, prompt_ids, vision_indices)
    length = embeds.shape[1]
    cache = jp.asarray(model.build_cache(length))
    positions = jp.arange(length, dtype="int32")[None]
    logits, _ = call_step(model, embeds, cache, 0, positions, per_layer)
    return logits
