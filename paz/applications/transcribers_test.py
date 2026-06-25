from pathlib import Path

import pytest

import paz
from paz.applications import TranscribeWhisper
from paz.models.foundation.whisper.tokenizer import ASSETS_DIR

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "examples" / "speech_to_text" / "whisper_models"
AUDIO_PATH = ROOT / "examples" / "speech_to_text" / "harvard.wav"


def has_assets():
    return (ASSETS_DIR / "tokenizer.json").exists()


def has_weights():
    return (WEIGHTS_DIR / "whisper_tiny_en" / "encoder.weights.h5").exists()


def has_vocabulary():
    return (ASSETS_DIR / "vocabulary.json").exists()


@pytest.mark.skipif(not has_assets(), reason="tokenizer assets not present")
def test_transcribe_whisper_builds_callable():
    transcribe = TranscribeWhisper("whisper_tiny_en", weights=None)
    assert callable(transcribe)


@pytest.mark.skipif(not (has_weights() and has_vocabulary()),
                    reason="whisper weights or vocabulary not present")
def test_transcribe_whisper_on_harvard():
    transcribe = TranscribeWhisper(
        "whisper_tiny_en", max_tokens=48, models_path=str(WEIGHTS_DIR))
    waveform, sample_rate = paz.audio.load(AUDIO_PATH)
    text = transcribe(waveform, sample_rate)
    assert "smell" in text.lower()
