"""One-time conversion: HuggingFace safetensors -> Keras .keras + .weights.h5.

Produces both .keras (architecture + weights) and .weights.h5 (weights only)
for each model half, plus a config.json and tokenizer assets.

Usage:
    python examples/gemma4/convert_hf_to_paz.py \
        --hf_path ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/...

Output directory (default: examples/gemma4/weights):
    config.json
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    embedding_step.keras
    embedding_step.weights.h5
    decoder_step.keras
    decoder_step.weights.h5
"""
import os
import sys

if os.environ.get("JAX_PLATFORMS") != "cpu":
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "KERAS_BACKEND": "jax"}
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import gc
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gemma4.hf_export import (
    extract_config_from_hf,
    load_hf_weights_into_paz,
    open_safetensors,
)
from examples.gemma4.inference import (
    Gemma4DecoderStep,
    Gemma4PerLayerEmbeddingStep,
)
from examples.gemma4.reference import save_config

WEIGHTS_DIR = Path(__file__).resolve().with_name("weights")


def convert(hf_path, output_dir):
    hf_path = Path(hf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_tokenizer_assets(hf_path, output_dir)
    config = extract_config_from_hf(hf_path / "config.json")
    sf_handle = open_safetensors(hf_path)

    if config.hidden_size_per_layer_input:
        save_embedding_step(config, sf_handle, output_dir)
        gc.collect()

    save_decoder_step(config, sf_handle, output_dir)
    gc.collect()

    config_path = output_dir / "config.json"
    save_config(config, config_path)
    print("Saved config to {}.".format(config_path), flush=True)


def copy_tokenizer_assets(hf_path, output_dir):
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    for name in names:
        source = hf_path / name
        if not source.exists():
            continue
        destination = output_dir / name
        shutil.copy2(str(source), str(destination))
        print("Copied {}.".format(destination), flush=True)


def save_embedding_step(config, sf_handle, output_dir):
    print("Building per-layer embedding model...", flush=True)
    embedding_step = Gemma4PerLayerEmbeddingStep(config)
    print("Loading per-layer embedding weights...", flush=True)
    load_hf_weights_into_paz(embedding_step, sf_handle, config)

    keras_path = output_dir / "embedding_step.keras"
    print("Saving {}...".format(keras_path), flush=True)
    embedding_step.save(str(keras_path))

    h5_path = output_dir / "embedding_step.weights.h5"
    print("Saving {}...".format(h5_path), flush=True)
    embedding_step.save_weights(str(h5_path))


def save_decoder_step(config, sf_handle, output_dir):
    print("Building decoder step model...", flush=True)
    decoder_step = Gemma4DecoderStep(config)
    print("Loading decoder weights...", flush=True)
    load_hf_weights_into_paz(decoder_step, sf_handle, config)

    keras_path = output_dir / "decoder_step.keras"
    print("Saving {}...".format(keras_path), flush=True)
    decoder_step.save(str(keras_path))

    h5_path = output_dir / "decoder_step.weights.h5"
    print("Saving {}...".format(h5_path), flush=True)
    decoder_step.save_weights(str(h5_path))


if __name__ == "__main__":
    description = (
        "Convert HuggingFace Gemma 4 safetensors to "
        "Keras .keras + .weights.h5")
    parser = argparse.ArgumentParser(description=description)
    add = parser.add_argument
    add("--hf_path", required=True,
        help="HF snapshot directory (model.safetensors + config.json)")
    add("--output_dir", default=str(WEIGHTS_DIR),
        help="Output directory (default: examples/gemma4/weights)")
    args = parser.parse_args()

    convert(args.hf_path, args.output_dir)
    print("Conversion complete. Output: {}".format(args.output_dir))
