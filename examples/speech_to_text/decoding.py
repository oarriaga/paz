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


def build_decoder_token_ids_and_padding_mask(token_ids):
    token_ids = ops.convert_to_tensor([token_ids], dtype="int32")
    padding_mask = ops.ones_like(token_ids, dtype="int32")
    return token_ids, padding_mask


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
