import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paz.applications import GenerateGemma4

# Until the weights are published, load them from the local example folder.
WEIGHTS_DIR = Path(__file__).with_name("weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma 4 text generation demo")
    add = parser.add_argument
    add("--model_name", default="gemma4_2b")
    add("--models_path", default=str(WEIGHTS_DIR))
    add("--prompt", default="The capital of Germany is")
    add("--max_tokens", default=32, type=int)
    args = parser.parse_args()
    generate = GenerateGemma4(
        args.model_name, args.max_tokens, models_path=args.models_path)
    print(generate(args.prompt))
