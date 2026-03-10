import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.speech_to_text.decoding import transcribe_whisper_base_en_wav
from examples.speech_to_text.model import (
    build_whisper_base_en_waveform_to_features_model,
)
from examples.speech_to_text.weights import (
    build_preset_loaded_whisper_base_en_decoder_model,
)
from examples.speech_to_text.weights import (
    build_preset_loaded_whisper_base_en_encoder_model,
)

audio_path = Path(__file__).with_name("harvard.wav")
frontend_model = build_whisper_base_en_waveform_to_features_model()
encoder_model = build_preset_loaded_whisper_base_en_encoder_model()
decoder_model = build_preset_loaded_whisper_base_en_decoder_model()
_, _, _, decoded_text = transcribe_whisper_base_en_wav(
    audio_path,
    frontend_model,
    encoder_model,
    decoder_model,
    max_generated_tokens=64,
)
print(decoded_text)
