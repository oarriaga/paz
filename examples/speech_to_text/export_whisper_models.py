import os

# This guide can only be run with the jax backend.
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.speech_to_text import weights as whisper_weights
from examples.speech_to_text.model2 import CONFIGS
from examples.speech_to_text.model2 import WHISPER_MODELS_DIR
from examples.speech_to_text.model2 import WhisperCrossCache
from examples.speech_to_text.model2 import WhisperDecoderStep
from examples.speech_to_text.model2 import WhisperEncoder


RUNTIME_MODEL_BUILDERS = {
    "encoder": WhisperEncoder,
    "cross_cache": WhisperCrossCache,
    "decoder_step": WhisperDecoderStep,
}


def export_whisper_models(variant_names=None, output_root=WHISPER_MODELS_DIR, dtype="float32"):  # fmt: skip
    variant_names = build_variant_names(variant_names)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for variant_name in variant_names:
        export_variant(variant_name, output_root, dtype)


def export_variant(variant_name, output_root, dtype):
    whisper_weights.require_whisper_preset_dir(variant_name)
    config = dict(CONFIGS[variant_name])
    variant_dir = Path(output_root) / variant_name
    reset_variant_dir(variant_dir)
    for model_kind, builder in RUNTIME_MODEL_BUILDERS.items():
        model = build_runtime_model(builder, variant_name, model_kind, config, dtype)
        whisper_weights.load_preset_weights(
            model, variant_name, dtype=dtype, model_kind=model_kind
        )
        model.save(str(build_model_path(variant_dir, model_kind)))
        model.save_weights(str(build_weights_path(variant_dir, model_kind)))


def build_runtime_model(builder, variant_name, model_kind, config, dtype):
    return builder(**config, dtype=dtype, name="{}_{}".format(variant_name, model_kind))  # fmt: skip


def build_model_path(variant_dir, model_kind):
    return Path(variant_dir) / "{}.keras".format(model_kind)


def build_weights_path(variant_dir, model_kind):
    return Path(variant_dir) / "{}.weights.h5".format(model_kind)


def reset_variant_dir(variant_dir):
    variant_dir = Path(variant_dir)
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)


def build_variant_names(variant_names):
    if variant_names:
        return tuple(variant_names)
    available_presets = whisper_weights.find_available_whisper_presets()
    return tuple(variant_name for variant_name, _ in available_presets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export local Whisper presets to standard Keras files."
    )
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--output_root", default=str(WHISPER_MODELS_DIR))
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()
    export_whisper_models(args.variants, args.output_root, args.dtype)
