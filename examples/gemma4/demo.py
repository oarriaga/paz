import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gemma4.causal_lm import Gemma4CausalLM
from examples.gemma4.decoding import KVDecoder, kv_decode
from examples.gemma4.decoding import extract_generated_ids
from examples.gemma4.inference import Gemma4DecoderStep
from examples.gemma4.reference import load_config
from examples.gemma4.tokenizer import Gemma4Tokenizer

MODELS_DIR = Path(__file__).resolve().with_name("gemma4_models")


def generate(prompt, models, tokenizer, max_tokens=64):
    step_model, config = models
    token_ids = tokenizer.tokenize(prompt)
    stop_id = tokenizer.end_token_id
    args = (step_model, config, token_ids, stop_id, max_tokens)
    all_ids = kv_decode(*args)
    generated = extract_generated_ids(
        all_ids, len(token_ids), stop_id)
    return tokenizer.detokenize(generated)


def build_models(model_dir):
    model_dir = Path(model_dir)
    config = load_config(model_dir / "config.json")
    weights_path = model_dir / "model.weights.h5"
    step = Gemma4DecoderStep(config)
    step.load_weights(str(weights_path))
    return step, config


def build_tokenizer(model_dir):
    proto_path = Path(model_dir) / "tokenizer.spm"
    return Gemma4Tokenizer(proto_path, add_bos=True)


if __name__ == "__main__":
    description = "Gemma 4 text generation demo"
    parser = argparse.ArgumentParser(description=description)
    default_dir = str(MODELS_DIR / "gemma4_2b")
    add = parser.add_argument
    add("--model_dir", default=default_dir)
    add("--prompt", default="The meaning of life is")
    add("--max_tokens", default=64, type=int)
    args = parser.parse_args()
    tokenizer = build_tokenizer(args.model_dir)
    models = build_models(args.model_dir)
    text = generate(args.prompt, models, tokenizer, args.max_tokens)
    print(text)
