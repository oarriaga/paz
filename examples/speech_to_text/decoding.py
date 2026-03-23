import jax
import jax.numpy as jp
from keras import ops

from examples.speech_to_text.tokenizer import find_special_token_id


def build_whisper_base_en_prompt_token_ids(tokenizer_config_path=None):
    # The local whisper_base_en tokenizer config has no language tokens.
    return [
        find_special_token_id(
            "<|startoftranscript|>", tokenizer_config_path
        ),
        find_special_token_id("<|transcribe|>", tokenizer_config_path),
        find_special_token_id("<|notimestamps|>", tokenizer_config_path),
    ]


def KVDecoder(
    decoder_step_model,
    prompt_token_ids,
    max_generated_tokens,
    max_decoder_sequence_length=448,
):
    max_decode_length = min(
        max_decoder_sequence_length,
        len(prompt_token_ids) + max_generated_tokens,
    )
    prompt_token_ids = jp.array(prompt_token_ids, dtype=jp.int32)
    prompt_length = len(prompt_token_ids)

    @jax.jit
    def decode(self_attention_cache, cross_attention_cache, stop_token_id):
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

    decode.max_decode_length = max_decode_length
    return decode


def decode_token_ids_with_kv_cache(
    decoder,
    cache_shape,
    cross_cache_model,
    encoder_output,
    stop_token_id,
):
    self_attention_cache = _build_self_attention_cache(
        cache_shape,
        decoder.max_decode_length,
        int(encoder_output.shape[0]),
    )
    cross_attention_cache = cross_cache_model(encoder_output)
    self_attention_cache = jp.asarray(self_attention_cache)
    cross_attention_cache = jp.asarray(cross_attention_cache)
    token_buffer, current_length = decoder(
        self_attention_cache,
        cross_attention_cache,
        jp.array(stop_token_id, dtype=jp.int32),
    )
    token_ids = ops.convert_to_numpy(token_buffer[0, :current_length]).tolist()
    return token_ids


def extract_text_token_ids(token_ids, prompt_length, stop_token_id):
    text_token_ids = token_ids[prompt_length:]
    if stop_token_id in text_token_ids:
        stop_index = text_token_ids.index(stop_token_id)
        text_token_ids = text_token_ids[:stop_index]
    return text_token_ids


def _build_self_attention_cache(
    cache_shape, max_decode_length, batch_size=1, dtype="float32"
):
    num_layers = int(cache_shape[1])
    num_heads = int(cache_shape[4])
    key_dim = int(cache_shape[5])
    return ops.zeros(
        (batch_size, num_layers, 2, max_decode_length, num_heads, key_dim),
        dtype=dtype,
    )
