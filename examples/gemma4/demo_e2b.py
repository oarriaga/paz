import os
import sys

# Force CPU-only JAX to avoid GPU OOM during graph construction.
# Must happen before any JAX/Keras initialization.
# if os.environ.get("JAX_PLATFORMS") != "cpu":
#     env = {**os.environ, "JAX_PLATFORMS": "cpu", "KERAS_BACKEND": "jax"}
#     os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax.numpy as jnp

from examples.gemma4.inference import (
    Gemma4PerLayerEmbeddingStep,
    Gemma4DecoderStep,
    build_empty_cache,
)
from examples.gemma4.reference import load_config
from examples.gemma4.tokenizer import Gemma4Tokenizer

WEIGHTS_DIR = Path(__file__).resolve().with_name("weights")


def build_tokenizer(weights_dir):
    tokenizer_path = Path(weights_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        message = "Expected tokenizer asset at '{}'".format(tokenizer_path)
        raise FileNotFoundError(message)
    return Gemma4Tokenizer(tokenizer_path)


def build_models(weights_dir):
    """Build and load both model halves from saved Keras weights.

    Loading strategy (memory budget ~12 GB):
      1. Build Gemma4PerLayerEmbeddingStep + load its 4.7 GB weight.
         Peak: ~9.4 GB (4.7 GB zeros + 4.7 GB from H5).
      2. Build Gemma4DecoderStep + load its 4.57 GB weights.
         Peak: ~10 GB (both models + one weight from H5).
    """
    weights_dir = Path(weights_dir)
    config = load_config(weights_dir / "config.json")

    print("Building per-layer embedding model...", flush=True)
    embedding_step = Gemma4PerLayerEmbeddingStep(config)
    h5_path = weights_dir / "embedding_step.weights.h5"
    print("Loading weights from {}...".format(h5_path), flush=True)
    embedding_step.load_weights(str(h5_path))
    gc.collect()

    print("Building decoder step model...", flush=True)
    decoder_step = Gemma4DecoderStep(config)
    h5_path = weights_dir / "decoder_step.weights.h5"
    print("Loading weights from {}...".format(h5_path), flush=True)
    decoder_step.load_weights(str(h5_path))
    gc.collect()

    return embedding_step, decoder_step, config


def tokenize_prompt(prompt, tokenizer):
    return tokenizer.tokenize_generation_prompt(prompt)


def generate(prompt, models, tokenizer, max_tokens=256):
    embedding_step, decoder_step, config = models
    token_ids = tokenize_prompt(prompt, tokenizer)
    stop_ids = set(tokenizer.get_stop_token_ids())
    max_len = len(token_ids) + max_tokens

    cache = jnp.array(build_empty_cache(config, max_len))

    # Warmup: fill KV cache for each prompt token except the last.
    for i in range(len(token_ids) - 1):
        token = jnp.array([[token_ids[i]]], dtype=jnp.int32)
        index = jnp.array([i], dtype=jnp.int32)
        per_layer = embedding_step([token])
        _, cache = decoder_step([token, cache, index, per_layer])

    # Decode: step from the last prompt token, greedy argmax.
    generated = []
    token = jnp.array([[token_ids[-1]]], dtype=jnp.int32)
    index = jnp.array([len(token_ids) - 1], dtype=jnp.int32)

    for step_i in range(max_tokens):
        per_layer = embedding_step([token])
        logits, cache = decoder_step([token, cache, index, per_layer])
        next_id = int(jnp.argmax(logits[0, 0]))
        if next_id in stop_ids:
            break
        generated.append(next_id)
        token = jnp.array([[next_id]], dtype=jnp.int32)
        index = index + 1
        if (step_i + 1) % 10 == 0:
            partial = tokenizer.detokenize(generated)
            print("  [{}] {}".format(step_i + 1, partial), flush=True)

    return tokenizer.detokenize(generated)


if __name__ == "__main__":
    description = "Gemma 4 E2B text generation demo"
    parser = argparse.ArgumentParser(description=description)
    add = parser.add_argument
    add(
        "--weights_dir",
        default=str(WEIGHTS_DIR),
        help="Directory with config.json + .weights.h5 files",
    )
    add("--prompt", default="The meaning of life is")
    add("--max_tokens", default=256, type=int)
    args = parser.parse_args()

    print("Loading tokenizer...", flush=True)
    tokenizer = build_tokenizer(args.weights_dir)

    print("Building models from {}...".format(args.weights_dir), flush=True)
    models = build_models(args.weights_dir)

    print("Generating (prints every 10 tokens)...\n", flush=True)
    text = generate(args.prompt, models, tokenizer, args.max_tokens)
    print("\n--- Final output ---")
    print(text)
