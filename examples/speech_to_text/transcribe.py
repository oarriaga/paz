import sys
from pathlib import Path

import numpy as np
from keras import ops

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.speech_to_text.audio import load
from examples.speech_to_text.audio import resample
from examples.speech_to_text.audio import to_float
from examples.speech_to_text.audio import to_mono
from examples.speech_to_text.decoding import (
    build_whisper_base_en_prompt_token_ids,
)
from examples.speech_to_text.decoding import (
    compute_argmax_token_id_from_decoder_model,
)
from examples.speech_to_text.decoding import decode_token_ids
from examples.speech_to_text.decoding import extract_text_token_ids
from examples.speech_to_text.model import (
    build_whisper_base_en_waveform_to_features_model,
)
from examples.speech_to_text.tokenizer import decode_whisper_token_ids
from examples.speech_to_text.tokenizer import find_special_token_id
from examples.speech_to_text.weights import (
    build_preset_loaded_whisper_base_en_decoder_model,
)
from examples.speech_to_text.weights import (
    build_preset_loaded_whisper_base_en_encoder_model,
)


def main(audio_path=None, max_generated_tokens=64):
    if audio_path is None:
        audio_path = Path(__file__).with_name("harvard.wav")
    frontend_model = build_whisper_base_en_waveform_to_features_model()
    encoder_model = build_preset_loaded_whisper_base_en_encoder_model()
    decoder_model = build_preset_loaded_whisper_base_en_decoder_model()
    _, _, _, decoded_text = transcribe_whisper_base_en_wav(
        audio_path,
        frontend_model,
        encoder_model,
        decoder_model,
        max_generated_tokens=max_generated_tokens,
    )
    print(decoded_text)
    return decoded_text


def transcribe_whisper_base_en_wav(
    wav_path,
    frontend_model,
    encoder_model,
    decoder_model,
    max_generated_tokens=64,
):
    sample_rate, waveform = preprocess_wav_for_whisper(wav_path)
    token_ids, generated_token_ids, decoded_text = (
        transcribe_whisper_base_en_waveform(
            waveform,
            frontend_model,
            encoder_model,
            decoder_model,
            max_generated_tokens=max_generated_tokens,
        )
    )
    return sample_rate, token_ids, generated_token_ids, decoded_text


def transcribe_whisper_base_en_waveform(
    waveform,
    frontend_model,
    encoder_model,
    decoder_model,
    max_generated_tokens=64,
):
    waveform_batch = build_single_waveform_batch(waveform)
    encoder_features = frontend_model(waveform_batch)
    encoder_output = encoder_model(encoder_features)
    prompt_token_ids = build_whisper_base_en_prompt_token_ids()
    stop_token_id = find_special_token_id("<|endoftext|>")

    def next_token_id_function(token_ids):
        return compute_argmax_token_id_from_decoder_model(
            decoder_model, encoder_output, token_ids
        )

    token_ids = decode_token_ids(
        next_token_id_function,
        prompt_token_ids,
        stop_token_id,
        max_generated_tokens=max_generated_tokens,
    )
    generated_token_ids = extract_text_token_ids(
        token_ids, len(prompt_token_ids), stop_token_id
    )
    decoded_text = decode_whisper_token_ids(generated_token_ids)
    return token_ids, generated_token_ids, decoded_text


def preprocess_wav_for_whisper(wav_path, expected_sample_rate=16000):
    waveform, sample_rate = load(wav_path)
    waveform = to_float(waveform)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sample_rate, expected_sample_rate)
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform = ops.convert_to_tensor(waveform, dtype="float32")
    return expected_sample_rate, waveform


def build_single_waveform_batch(waveform):
    if len(waveform.shape) == 1:
        waveform = ops.expand_dims(waveform, axis=0)
    return waveform


if __name__ == "__main__":
    audio_path = None if len(sys.argv) < 2 else sys.argv[1]
    main(audio_path)
