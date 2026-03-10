from pathlib import Path

import pytest

from examples.speech_to_text import transcribe
from examples.speech_to_text.weights import build_missing_whisper_preset_message
from examples.speech_to_text.weights import find_whisper_base_en_preset_dir


def test_transcribe_main_returns_harvard_text(clear_keras_session):
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    audio_path = Path(__file__).with_name("harvard.wav")
    assert transcribe.main(audio_path) == (
        " The stale smell of old-beer lingers. It takes heat to bring "
        "out the odor. A cold dip restores health and zest. A salt "
        "pickle tastes fine with ham. Tacos al pastor are my favorite. "
        "A zestful food is the hot cross bun."
    )
