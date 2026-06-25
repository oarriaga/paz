import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paz
from paz.applications import TranscribeWhisper
from paz.models.foundation.whisper.configuration import CONFIGS

# Until the weights are published, load them from the local example folder.
WEIGHTS_DIR = Path(__file__).with_name("whisper_models")


if __name__ == "__main__":
    description = "Whisper speech-to-text demo"
    parser = argparse.ArgumentParser(description=description)
    default_audio = str(Path(__file__).with_name("harvard.wav"))
    add = parser.add_argument
    add("--audio_path", default=default_audio)
    add("--model_name", default="whisper_base_en", choices=list(CONFIGS))
    add("--max_tokens", default=64, type=int)
    args = parser.parse_args()
    transcribe = TranscribeWhisper(
        args.model_name, args.max_tokens, models_path=str(WEIGHTS_DIR))
    waveform, sample_rate = paz.audio.load(args.audio_path)
    print(transcribe(waveform, sample_rate))
