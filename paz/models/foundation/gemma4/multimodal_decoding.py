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
import jax.numpy as jp
import numpy as np
from keras import ops

from paz import call_stateless, snapshot_variables

from paz.models.foundation.gemma4.inference import build_empty_cache
from paz.models.foundation.gemma4.sampling import sample_logits


def as_token(token):
    return jp.array([[int(token)]], dtype="int32")


def build_prompt_rows(step_model, vision_embeddings, prompt_ids,
                      vision_indices, per_layer_step):
    embedding = step_model.get_layer("token_embedding")
    image_position = {}
    for image_index, position in enumerate(vision_indices):
        image_position[int(position)] = image_index
    embeds, per_layers = [], []
    for index, token in enumerate(prompt_ids):
        if index in image_position:
            embeds.append(jp.asarray(vision_embeddings[image_position[index]]))
            per_layers.append(per_layer_row(per_layer_step, 0))
        else:
            embeds.append(embedding(as_token(token))[0, 0])
            per_layers.append(per_layer_row(per_layer_step, token))
    input_embeddings = jp.stack(embeds)[None]
    per_layer = None
    if per_layer_step is not None:
        per_layer = jp.stack(per_layers)[None]
    return input_embeddings, per_layer


def per_layer_row(per_layer_step, token):
    # Image positions reuse the pad-token (id 0) per-layer embedding, matching
    # keras_hub which zeroes the per-layer token ids at vision positions.
    if per_layer_step is None:
        return None
    return per_layer_step(as_token(token))[0, 0]


def call_step(step_model, embeds, cache, start, positions, per_layer):
    index = jp.reshape(jp.asarray(start, dtype="int32"), (1,))
    inputs = [embeds, cache, index, positions]
    if per_layer is not None:
        inputs.append(per_layer)
    return step_model(inputs)


def generate(step_model, per_layer_step, vision_embeddings, config,
             prompt_ids, vision_indices, stop_id, max_tokens):
    """Single-sequence jitted greedy generation."""
    return run_cached_generate(
        step_model, per_layer_step, vision_embeddings, config, prompt_ids,
        vision_indices, stop_id, max_tokens, jax.random.PRNGKey(0),
        select_greedy_token)


def generate_eager(step_model, per_layer_step, vision_embeddings, config,
                   prompt_ids, vision_indices, stop_id, max_tokens):
    """Eager parallel prefill + Python greedy decode loop.

    Prefills the whole prompt in ONE forward, then decodes token-by-token. The
    full-program jit in `generate` constant-folds the multi-GB embedding tables
    into the executable, which a modest host cannot afford alongside the
    resident weights; keras compiles each step, so this is equally fast for
    short outputs while staying within budget on the real E2B/E4B weights.
    """
    embeds, per_layer = build_prompt_rows(
        step_model, vision_embeddings, prompt_ids, vision_indices,
        per_layer_step)
    length = embeds.shape[1]
    cache = jp.asarray(build_empty_cache(config, length + max_tokens))
    positions = jp.arange(length, dtype="int32")[None]
    logits, cache = call_step(step_model, embeds, cache, 0, positions,
                              per_layer)
    token = int(jp.argmax(logits[0, -1]))
    embedding = step_model.get_layer("token_embedding")
    out, index = [token], length
    while token != stop_id and len(out) < max_tokens:
        per_layer = decode_per_layer(per_layer_step, token)
        logits, cache = call_step(
            step_model, embedding(as_token(token)), cache, index,
            jp.array([[index]], dtype="int32"), per_layer)
        token = int(jp.argmax(logits[0, -1]))
        out.append(token)
        index += 1
    return trim_to_stop(out, stop_id)


def decode_per_layer(per_layer_step, token):
    if per_layer_step is None:
        return None
    return per_layer_step(as_token(token))


def generate_sample(step_model, per_layer_step, vision_embeddings, config,
                    prompt_ids, vision_indices, stop_id, max_tokens, key,
                    sampling):
    """Single-sequence jitted sampling generation seeded by `key`."""
    return run_cached_generate(
        step_model, per_layer_step, vision_embeddings, config, prompt_ids,
        vision_indices, stop_id, max_tokens, key, build_token_sampler(sampling))


def select_greedy_token(logits, key):
    return jp.argmax(logits[0]).astype("int32")


def build_token_sampler(sampling):
    return lambda logits, key: sample_logits(logits, key, sampling)[0]


def run_cached_generate(step_model, per_layer_step, vision_embeddings, config,
                        prompt_ids, vision_indices, stop_id, max_tokens, key,
                        select):
    embeds, per_layer = build_prompt_rows(
        step_model, vision_embeddings, prompt_ids, vision_indices,
        per_layer_step)
    length = embeds.shape[1]
    max_length = length + max_tokens
    cache = jp.asarray(build_empty_cache(config, max_length))
    positions = jp.arange(length, dtype="int32")[None]
    logits, cache = call_step(
        step_model, embeds, cache, 0, positions, per_layer)
    key, first_key = jax.random.split(key)
    first = select(logits[:, -1], first_key)
    decode = build_decode_loop(
        step_model, per_layer_step, max_length, max_tokens, select)
    variables = (
        snapshot_variables(step_model),
        snapshot_variables(step_model.get_layer("token_embedding")),
        snapshot_variables(per_layer_step))
    buffer, count = decode(cache, first, jp.array(length, "int32"),
                           jp.array(stop_id, "int32"), key, variables)
    generated = ops.convert_to_numpy(buffer[:int(count)]).tolist()
    return trim_to_stop(generated, stop_id)


def trim_to_stop(tokens, stop_id):
    if stop_id in tokens:
        tokens = tokens[:tokens.index(stop_id)]
    return tokens


def build_decode_loop(step_model, per_layer_step, max_length, max_tokens,
                      select):
    embedding = step_model.get_layer("token_embedding")

    @jax.jit
    def decode(cache, first_token, start_index, stop_id, key, variables):
        buffer = jp.zeros((max_tokens,), dtype="int32").at[0].set(first_token)
        count = jp.array(1, dtype="int32")
        state = (buffer, first_token, start_index, cache, count,
                 first_token == stop_id, key)
        cont = build_should_continue(max_tokens, max_length)
        step = build_decode_step(
            step_model, embedding, per_layer_step, stop_id, select, variables)
        buffer, _, _, _, count, _, _ = jax.lax.while_loop(cont, step, state)
        return buffer, count

    return decode


def build_should_continue(max_tokens, max_length):
    def check(state):
        _, _, index, _, count, finished, _ = state
        return (~finished) & (count < max_tokens) & (index < max_length)
    return check


def build_decode_step(step_model, embedding, per_layer_step, stop_id, select,
                      variables):
    step_vars, embedding_vars, per_layer_vars = variables

    def step(state):
        buffer, token, index, cache, count, _, key = state
        token2d = jp.reshape(token, (1, 1))
        positions = jp.reshape(index, (1, 1)).astype("int32")
        embeds = call_stateless(embedding, embedding_vars, token2d)
        inputs = [embeds, cache, jp.reshape(index, (1,)).astype("int32"),
                  positions]
        if per_layer_vars is not None:
            inputs.append(
                call_stateless(per_layer_step, per_layer_vars, token2d))
        logits, cache = call_stateless(step_model, step_vars, inputs)
        key, step_key = jax.random.split(key)
        next_id = select(logits[:, 0, :], step_key)
        buffer = buffer.at[count].set(next_id)
        finished = next_id == stop_id
        return (buffer, next_id, index + 1, cache, count + 1, finished, key)
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
    prompts = jp.asarray(prompts, dtype="int32")
    batch, length = prompts.shape
    embeds, per_layer = build_text_batch_rows(
        step_model, per_layer_step, prompts)
    cache = jp.asarray(build_empty_cache(config, length + max_tokens, batch))
    positions = jp.arange(length, dtype="int32")[None]
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
            jp.full((1, 1), index, dtype="int32"), per_layer)
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
    cache = jp.asarray(build_empty_cache(config, length))
    positions = jp.arange(length, dtype="int32")[None]
    logits, _ = call_step(step_model, embeds, cache, 0, positions, per_layer)
    return logits
