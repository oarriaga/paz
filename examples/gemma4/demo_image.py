import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paz.applications import DescribeImageGemma4


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma 4 image captioning demo")
    add = parser.add_argument
    add("--image", required=True)
    add("--question", default="Describe this image.")
    add("--model_name", default="gemma4_2b")
    # Default downloads the published weights; pass a local dir to override.
    add("--models_path", default=None)
    add("--max_tokens", default=32, type=int)
    args = parser.parse_args()
    describe = DescribeImageGemma4(
        args.model_name, args.max_tokens, models_path=args.models_path)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    print(describe(image, args.question))
