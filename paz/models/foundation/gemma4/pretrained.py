import json
from collections import namedtuple
from pathlib import Path

from paz.models.foundation.gemma4.configuration import load_config
from paz.models.foundation.gemma4.inference import Gemma4MultimodalDecoderStep
from paz.models.foundation.gemma4.inference import Gemma4PerLayerEmbeddingStep
from paz.models.foundation.gemma4.vision import VisionEncoderArgs
from paz.models.foundation.gemma4.vision import build_vision_encoder

GEMMA4_WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.24/"  # fmt: skip
Gemma4Models = namedtuple(
    "Gemma4", "config decoder_step per_layer_step vision_encoder")


def Gemma4(model_name="gemma4_2b", weights="paz", models_path=None):
    model_dir = resolve_gemma4_dir(model_name, models_path)
    config = load_config(model_dir / "config.json")
    decoder_step = Gemma4MultimodalDecoderStep(config)
    per_layer_step = build_per_layer_step(config)
    vision_encoder = build_vision_encoder_from_dir(model_dir)
    if weights is not None:
        decoder_step.load_weights(str(model_dir / "decoder_step.weights.h5"))
        load_optional_weights(model_dir, per_layer_step, vision_encoder)
    return Gemma4Models(config, decoder_step, per_layer_step, vision_encoder)


def resolve_gemma4_dir(model_name, models_path):
    if models_path is not None:
        return Path(models_path)
    message = ("Gemma4 weights for '{}' are not hosted yet; pass models_path "
               "to a local weights directory.").format(model_name)
    raise ValueError(message)


def build_per_layer_step(config):
    if not config.hidden_size_per_layer_input:
        return None
    return Gemma4PerLayerEmbeddingStep(config)


def build_vision_encoder_from_dir(model_dir):
    path = Path(model_dir) / "vision_config.json"
    if not path.exists():
        return None
    with open(str(path), encoding="utf-8") as file:
        config = VisionEncoderArgs(**json.load(file))
    return build_vision_encoder(config)


def load_optional_weights(model_dir, per_layer_step, vision_encoder):
    if per_layer_step is not None:
        per_layer_step.load_weights(
            str(model_dir / "embedding_step.weights.h5"))
    if vision_encoder is not None:
        vision_encoder.load_weights(
            str(model_dir / "vision_encoder.weights.h5"))
