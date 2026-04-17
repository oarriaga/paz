import gc
import os
import jax

jax.config.update("jax_platform_name", "cpu")

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gemma4.configuration import load_config
from examples.gemma4.inference import (
    Gemma4DecoderStep,
    Gemma4PerLayerEmbeddingStep,
    build_empty_cache,
)
from examples.gemma4.tokenizer import Gemma4Tokenizer

WEIGHTS_DIR = Path(__file__).resolve().with_name("weights")


def build_tokenizer(weights_dir):
    path = Path(weights_dir) / "tokenizer.json"
    if not path.exists():
        message = "Expected tokenizer asset at '{}'".format(path)
        raise FileNotFoundError(message)
    return Gemma4Tokenizer(path)


def build_models(weights_dir):
    weights_dir = Path(weights_dir)
    config = load_config(weights_dir / "config.json")
    embedding_model = Gemma4PerLayerEmbeddingStep(config)
    embedding_model.load_weights(str(weights_dir / "embedding_step.weights.h5"))
    gc.collect()
    step_model = Gemma4DecoderStep(config)
    step_model.load_weights(str(weights_dir / "decoder_step.weights.h5"))
    gc.collect()
    return embedding_model, step_model, config


def generate(prompt, models, tokenizer, max_tokens=256):
    embedding_model, step_model, config = models
    token_ids = tokenizer.tokenize_generation_prompt(prompt)
    stop_ids = set(tokenizer.get_stop_token_ids())
    max_len = len(token_ids) + max_tokens
    cache = jnp.array(build_empty_cache(config, max_len))
    for index in range(len(token_ids) - 1):
        token = jnp.array([[token_ids[index]]], dtype=jnp.int32)
        cache_index = jnp.array([index], dtype=jnp.int32)
        per_layer = embedding_model([token])
        _, cache = step_model([token, cache, cache_index, per_layer])
    generated = []
    token = jnp.array([[token_ids[-1]]], dtype=jnp.int32)
    cache_index = jnp.array([len(token_ids) - 1], dtype=jnp.int32)
    for _ in range(max_tokens):
        per_layer = embedding_model([token])
        logits, cache = step_model([token, cache, cache_index, per_layer])
        next_id = int(jnp.argmax(logits[0, 0]))
        if next_id in stop_ids:
            break
        generated.append(next_id)
        token = jnp.array([[next_id]], dtype=jnp.int32)
        cache_index = cache_index + 1
    return tokenizer.detokenize(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma 4 E2B text generation demo")
    add = parser.add_argument
    add("--weights_dir", default=str(WEIGHTS_DIR))
    add("--prompt", default="The capital of Germany is")
    add("--max_tokens", default=16, type=int)
    args = parser.parse_args()
    tokenizer = build_tokenizer(args.weights_dir)
    models = build_models(args.weights_dir)
    print(generate(args.prompt, models, tokenizer, args.max_tokens))
