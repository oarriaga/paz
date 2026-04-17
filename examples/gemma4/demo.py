import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gemma4.configuration import load_config
from examples.gemma4.decoding import extract_generated_ids, kv_decode
from examples.gemma4.inference import Gemma4DecoderStep
from examples.gemma4.tokenizer import Gemma4Tokenizer

MODELS_DIR = Path(__file__).resolve().with_name("weights")


def generate(prompt, models, tokenizer, max_tokens=64):
    step_model, config = models
    token_ids = tokenizer.tokenize(prompt)
    stop_id = tokenizer.end_token_id
    all_ids = kv_decode(step_model, config, token_ids, stop_id, max_tokens)
    new_ids = extract_generated_ids(all_ids, len(token_ids), stop_id)
    return tokenizer.detokenize(new_ids)


def build_models(model_dir):
    model_dir = Path(model_dir)
    config = load_config(model_dir / "config.json")
    if config.hidden_size_per_layer_input:
        message = "Split Gemma4 weights must use examples/gemma4/demo_e2b.py."
        raise ValueError(message)
    step_model = Gemma4DecoderStep(config)
    step_model.load_weights(str(build_weights_path(model_dir)))
    return step_model, config


def build_weights_path(model_dir):
    path = Path(model_dir) / "model.weights.h5"
    if path.exists():
        return path
    return Path(model_dir) / "decoder_step.weights.h5"


def build_tokenizer(model_dir):
    path = Path(model_dir) / "tokenizer.json"
    return Gemma4Tokenizer(path, add_bos=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma 4 text generation demo")
    add = parser.add_argument
    add("--model_dir", default=str(MODELS_DIR))
    add("--prompt", default="The meaning of life is")
    add("--max_tokens", default=64, type=int)
    args = parser.parse_args()
    tokenizer = build_tokenizer(args.model_dir)
    models = build_models(args.model_dir)
    print(generate(args.prompt, models, tokenizer, args.max_tokens))
