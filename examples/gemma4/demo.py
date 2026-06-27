import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paz.applications import GenerateGemma4


def chat(generate):
    print("Type a prompt and press Enter. Submit an empty line to quit.")
    while True:
        prompt = input("you> ").strip()
        if prompt == "":
            break
        print("gemma>", generate(prompt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma 4 text chat demo")
    add = parser.add_argument
    add("--model_name", default="gemma4_2b")
    # Default downloads the published weights; pass a local dir to override.
    add("--models_path", default=None)
    add("--max_tokens", default=64, type=int)
    args = parser.parse_args()
    generate = GenerateGemma4(
        args.model_name, args.max_tokens, models_path=args.models_path)
    chat(generate)
