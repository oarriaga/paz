import numpy as np
from keras import ops

from examples.speech_to_text.audio import load_waveform_from_wav
from examples.speech_to_text.tokenizer import decode_whisper_token_ids
from examples.speech_to_text.tokenizer import find_special_token_id


def build_whisper_base_en_decode_prompt(tokenizer_config_path=None):
    start_of_transcript = find_special_token_id(
        "<|startoftranscript|>",
        tokenizer_config_path,
    )
    transcribe = find_special_token_id("<|transcribe|>", tokenizer_config_path)
    no_timestamps = find_special_token_id(
        "<|notimestamps|>",
        tokenizer_config_path,
    )
    return [start_of_transcript, transcribe, no_timestamps]


def build_single_waveform_batch(waveform):
    rank = len(waveform.shape)
    if rank == 1:
        return ops.expand_dims(waveform, axis=0)
    if rank == 2 and waveform.shape[0] == 1:
        return waveform
    raise ValueError("Expected a single waveform with rank 1 or batch size 1.")


def build_decoder_inputs(token_ids):
    token_ids = np.asarray(token_ids, dtype="int32")
    token_ids = np.expand_dims(token_ids, axis=0)
    padding_mask = np.ones_like(token_ids, dtype="int32")
    decoder_token_ids = ops.convert_to_tensor(token_ids, dtype="int32")
    decoder_padding_mask = ops.convert_to_tensor(padding_mask, dtype="int32")
    return decoder_token_ids, decoder_padding_mask


def compute_next_token_id(logits, current_length):
    next_token_logits = logits[:, current_length - 1, :]
    next_token_id = ops.argmax(next_token_logits, axis=-1)
    next_token_id = ops.convert_to_numpy(next_token_id)
    return int(next_token_id[0])


def compute_decoder_logits(decoder_model, encoder_output, token_ids):
    decoder_token_ids, decoder_padding_mask = build_decoder_inputs(token_ids)
    decoder_output = decoder_model(
        [decoder_token_ids, decoder_padding_mask, encoder_output]
    )
    embedding_layer = decoder_model.get_layer("decoder_token_and_position_embedding")
    return embedding_layer.token_embedding(decoder_output, reverse=True)


def compute_model_next_token_id(decoder_model, encoder_output, token_ids):
    logits = compute_decoder_logits(decoder_model, encoder_output, token_ids)
    return compute_next_token_id(logits, len(token_ids))


def compute_greedy_decode(
    next_token_function,
    initial_token_ids,
    stop_token_id,
    max_generated_tokens=64,
    max_decoder_sequence_length=448,
):
    token_ids = list(initial_token_ids)
    while len(token_ids) < max_decoder_sequence_length:
        generated_length = len(token_ids) - len(initial_token_ids)
        if generated_length >= max_generated_tokens:
            break
        next_token_id = int(next_token_function(token_ids))
        token_ids.append(next_token_id)
        if next_token_id == stop_token_id:
            break
    return token_ids


def build_text_token_ids(token_ids, prompt_length, stop_token_id):
    token_ids = list(token_ids[prompt_length:])
    if stop_token_id in token_ids:
        stop_index = token_ids.index(stop_token_id)
        token_ids = token_ids[:stop_index]
    return token_ids


def transcribe_whisper_base_en_wav(
    wav_path,
    frontend_model,
    encoder_model,
    decoder_model,
    vocabulary_filepath=None,
    tokenizer_config_path=None,
    max_generated_tokens=64,
):
    sample_rate, waveform = load_waveform_from_wav(wav_path)
    token_ids, generated_token_ids, decoded_text = (
        transcribe_whisper_base_en_waveform(
            waveform,
            frontend_model,
            encoder_model,
            decoder_model,
            vocabulary_filepath,
            tokenizer_config_path,
            max_generated_tokens,
        )
    )
    return sample_rate, token_ids, generated_token_ids, decoded_text


def transcribe_whisper_base_en_waveform(
    waveform,
    frontend_model,
    encoder_model,
    decoder_model,
    vocabulary_filepath=None,
    tokenizer_config_path=None,
    max_generated_tokens=64,
):
    waveform = build_single_waveform_batch(waveform)
    encoder_features = frontend_model(waveform)
    encoder_output = encoder_model(encoder_features)
    prompt_token_ids = build_whisper_base_en_decode_prompt(
        tokenizer_config_path
    )
    stop_token_id = find_special_token_id("<|endoftext|>", tokenizer_config_path)

    def next_token_function(token_ids):
        return compute_model_next_token_id(decoder_model, encoder_output, token_ids)

    token_ids = compute_greedy_decode(
        next_token_function,
        prompt_token_ids,
        stop_token_id,
        max_generated_tokens,
        448,
    )
    generated_token_ids = build_text_token_ids(
        token_ids,
        len(prompt_token_ids),
        stop_token_id,
    )
    decoded_text = decode_whisper_token_ids(
        generated_token_ids,
        vocabulary_filepath,
        tokenizer_config_path,
        skip_special_tokens=True,
    )
    return token_ids, generated_token_ids, decoded_text
