import jax
import jax.numpy as jp
import numpy as np
from keras import ops

from examples.speech_to_text.tokenizer import find_special_token_id

COMPILED_DECODE_FUNCTIONS = {}
COMPILED_KV_DECODE_FUNCTIONS = {}


def build_whisper_base_en_prompt_token_ids(tokenizer_config_path=None):
    # The local whisper_base_en tokenizer config has no language tokens.
    return [
        find_special_token_id(
            "<|startoftranscript|>", tokenizer_config_path
        ),
        find_special_token_id("<|transcribe|>", tokenizer_config_path),
        find_special_token_id("<|notimestamps|>", tokenizer_config_path),
    ]


def build_decoder_token_ids_and_padding_mask(token_ids):
    token_ids = ops.convert_to_tensor([token_ids], dtype="int32")
    padding_mask = ops.ones_like(token_ids, dtype="int32")
    return token_ids, padding_mask


def build_token_id_buffer(initial_token_ids, max_decode_length):
    token_buffer = np.zeros((1, max_decode_length), dtype="int32")
    token_buffer[0, : len(initial_token_ids)] = initial_token_ids
    return ops.convert_to_tensor(token_buffer, dtype="int32")


def compute_argmax_token_id(logits, current_length):
    last_index = current_length - 1
    last_logits = logits[:, last_index, :]
    return int(ops.convert_to_numpy(ops.argmax(last_logits, axis=-1))[0])


def compute_decoder_hidden_states(decoder_model, encoder_output, token_ids):
    decoder_token_ids, padding_mask = build_decoder_token_ids_and_padding_mask(
        token_ids
    )
    return decoder_model([decoder_token_ids, padding_mask, encoder_output])


def compute_tied_decoder_logits(decoder_model, decoder_hidden_states):
    token_embedding = decoder_model.get_layer("decoder_token_embedding")
    return token_embedding(decoder_hidden_states, reverse=True)


def compute_argmax_token_id_from_decoder_model(
    decoder_model, encoder_output, token_ids
):
    decoder_hidden_states = compute_decoder_hidden_states(
        decoder_model, encoder_output, token_ids
    )
    logits = compute_tied_decoder_logits(decoder_model, decoder_hidden_states)
    return compute_argmax_token_id(logits, len(token_ids))


def compute_argmax_token_id_from_model(model, encoder_features, token_ids):
    decoder_token_ids, padding_mask = build_decoder_token_ids_and_padding_mask(
        token_ids
    )
    logits = model([encoder_features, decoder_token_ids, padding_mask])[2]
    return compute_argmax_token_id(logits, len(token_ids))


def find_compiled_decode_function(decoder_model, max_decode_length):
    cache_key = (id(decoder_model), max_decode_length)
    if cache_key not in COMPILED_DECODE_FUNCTIONS:
        COMPILED_DECODE_FUNCTIONS[cache_key] = build_compiled_decode_function(
            decoder_model, max_decode_length
        )
    return COMPILED_DECODE_FUNCTIONS[cache_key]


def build_self_attention_cache(
    decoder_step_model, max_decode_length, batch_size=1, dtype="float32"
):
    cache_shape = decoder_step_model.input_shape[1]
    num_layers = int(cache_shape[1])
    num_heads = int(cache_shape[4])
    key_dim = int(cache_shape[5])
    return ops.zeros(
        (batch_size, num_layers, 2, max_decode_length, num_heads, key_dim),
        dtype=dtype,
    )


def find_compiled_kv_decode_function(
    decoder_step_model, prompt_token_ids, max_decode_length
):
    cache_key = (
        id(decoder_step_model),
        tuple(prompt_token_ids),
        max_decode_length,
    )
    if cache_key not in COMPILED_KV_DECODE_FUNCTIONS:
        COMPILED_KV_DECODE_FUNCTIONS[cache_key] = (
            build_compiled_kv_decode_function(
                decoder_step_model, prompt_token_ids, max_decode_length
            )
        )
    return COMPILED_KV_DECODE_FUNCTIONS[cache_key]


def build_compiled_kv_decode_function(
    decoder_step_model, prompt_token_ids, max_decode_length
):
    prompt_token_ids = jp.array(prompt_token_ids, dtype=jp.int32)
    prompt_length = len(prompt_token_ids)

    @jax.jit
    def decode(
        self_attention_cache,
        cross_attention_cache,
        stop_token_id,
        max_generated_tokens,
    ):
        token_buffer = jp.zeros((1, max_decode_length), dtype=jp.int32)
        token_buffer = token_buffer.at[0, :prompt_length].set(prompt_token_ids)

        def warmup_step(index, self_attention_cache):
            token_id = jp.reshape(prompt_token_ids[index], (1, 1))
            _, self_attention_cache = decoder_step_model(
                [
                    token_id,
                    self_attention_cache,
                    cross_attention_cache,
                    jp.array([index], dtype=jp.int32),
                ]
            )
            return self_attention_cache

        if prompt_length > 1:
            self_attention_cache = jax.lax.fori_loop(
                0, prompt_length - 1, warmup_step, self_attention_cache
            )

        def should_continue(state):
            _, _, current_index, _, num_generated_tokens, finished = state
            return (
                (~finished)
                & (num_generated_tokens < max_generated_tokens)
                & (current_index + 1 < max_decode_length)
            )

        def next_state(state):
            (
                token_buffer,
                current_token_id,
                current_index,
                self_attention_cache,
                num_generated_tokens,
                _,
            ) = state
            logits, self_attention_cache = decoder_step_model(
                [
                    current_token_id,
                    self_attention_cache,
                    cross_attention_cache,
                    jp.array([current_index], dtype=jp.int32),
                ]
            )
            next_token_id = jp.argmax(logits[:, 0, :], axis=-1).astype(jp.int32)
            next_index = current_index + 1
            token_buffer = token_buffer.at[0, next_index].set(next_token_id[0])
            current_token_id = jp.expand_dims(next_token_id, axis=-1)
            finished = next_token_id[0] == stop_token_id
            return (
                token_buffer,
                current_token_id,
                next_index,
                self_attention_cache,
                num_generated_tokens + 1,
                finished,
            )

        token_buffer, _, current_index, _, _, _ = jax.lax.while_loop(
            should_continue,
            next_state,
            (
                token_buffer,
                jp.reshape(prompt_token_ids[prompt_length - 1], (1, 1)),
                jp.array(prompt_length - 1, dtype=jp.int32),
                self_attention_cache,
                jp.array(0, dtype=jp.int32),
                jp.array(False),
            ),
        )
        return token_buffer, current_index + 1

    return decode


def build_compiled_decode_function(decoder_model, max_decode_length):
    token_embedding = decoder_model.get_layer("decoder_token_embedding")
    token_positions = jp.arange(max_decode_length, dtype=jp.int32)

    @jax.jit
    def decode(encoder_output, token_buffer, prompt_length, stop_token_id):
        def should_continue(state):
            _, current_length, finished = state
            return (current_length < max_decode_length) & (~finished)

        def next_state(state):
            token_buffer, current_length, _ = state
            padding_mask = jp.expand_dims(
                (token_positions < current_length).astype(jp.int32), axis=0
            )
            decoder_hidden_states = decoder_model(
                [token_buffer, padding_mask, encoder_output]
            )
            logits = token_embedding(decoder_hidden_states, reverse=True)
            next_token_id = jp.argmax(
                logits[:, current_length - 1, :], axis=-1
            )[0].astype(jp.int32)
            token_buffer = token_buffer.at[0, current_length].set(
                next_token_id
            )
            finished = next_token_id == stop_token_id
            return token_buffer, current_length + 1, finished

        return jax.lax.while_loop(
            should_continue,
            next_state,
            (token_buffer, prompt_length, jp.array(False)),
        )

    return decode


def decode_token_ids_with_jax_loop(
    decoder_model,
    encoder_output,
    initial_token_ids,
    stop_token_id,
    max_generated_tokens=64,
    max_decoder_sequence_length=448,
):
    max_decode_length = min(
        max_decoder_sequence_length,
        len(initial_token_ids) + max_generated_tokens,
    )
    token_buffer = build_token_id_buffer(initial_token_ids, max_decode_length)
    decode = find_compiled_decode_function(decoder_model, max_decode_length)
    token_buffer, current_length, _ = decode(
        encoder_output,
        token_buffer,
        jp.array(len(initial_token_ids), dtype=jp.int32),
        jp.array(stop_token_id, dtype=jp.int32),
    )
    token_ids = ops.convert_to_numpy(token_buffer[0, :current_length]).tolist()
    return token_ids


def decode_token_ids_with_kv_cache(
    decoder_step_model,
    cross_cache_model,
    encoder_output,
    initial_token_ids,
    stop_token_id,
    max_generated_tokens=64,
    max_decoder_sequence_length=448,
):
    max_decode_length = min(
        max_decoder_sequence_length,
        len(initial_token_ids) + max_generated_tokens,
    )
    self_attention_cache = build_self_attention_cache(
        decoder_step_model,
        max_decode_length,
        int(encoder_output.shape[0]),
    )
    cross_attention_cache = cross_cache_model(encoder_output)
    decode = find_compiled_kv_decode_function(
        decoder_step_model, initial_token_ids, max_decode_length
    )
    token_buffer, current_length = decode(
        self_attention_cache,
        cross_attention_cache,
        jp.array(stop_token_id, dtype=jp.int32),
        jp.array(max_generated_tokens, dtype=jp.int32),
    )
    token_ids = ops.convert_to_numpy(token_buffer[0, :current_length]).tolist()
    return token_ids


def decode_token_ids(
    next_token_id_function,
    initial_token_ids,
    stop_token_id,
    max_generated_tokens=64,
    max_decoder_sequence_length=448,
):
    token_ids = list(initial_token_ids)
    num_generated_tokens = 0
    while num_generated_tokens < max_generated_tokens:
        if len(token_ids) >= max_decoder_sequence_length:
            break
        next_token_id = next_token_id_function(token_ids)
        token_ids.append(next_token_id)
        num_generated_tokens += 1
        if next_token_id == stop_token_id:
            break
    return token_ids


def extract_text_token_ids(token_ids, prompt_length, stop_token_id):
    text_token_ids = token_ids[prompt_length:]
    if stop_token_id in text_token_ids:
        stop_index = text_token_ids.index(stop_token_id)
        text_token_ids = text_token_ids[:stop_index]
    return text_token_ids
