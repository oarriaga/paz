import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

from keras.utils import get_file

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paz
from paz.applications import DescribeImageGemma4

SAMPLE_IMAGE_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.9.1/image_with_everyday_classes.jpg"  # fmt: skip


def download_sample_image():
    name = os.path.basename(SAMPLE_IMAGE_URL)
    path = get_file(name, SAMPLE_IMAGE_URL, cache_subdir="paz/images")
    return paz.image.load(path)


def chat_about_image(describe, image):
    print("Ask about the image. Submit an empty line to quit.")
    while True:
        question = input("you> ").strip()
        if question == "":
            break
        print("gemma>", describe(image, question))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma 4 image chat demo")
    add = parser.add_argument
    add("--model_name", default="gemma4_2b")
    # Default downloads the published weights; pass a local dir to override.
    add("--models_path", default=None)
    # Default downloads a sample image; pass a local path to override.
    add("--image", default=None)
    add("--max_tokens", default=64, type=int)
    args = parser.parse_args()
    describe = DescribeImageGemma4(
        args.model_name, args.max_tokens, models_path=args.models_path)
    if args.image is None:
        image = download_sample_image()
    else:
        image = paz.image.load(args.image)
    chat_about_image(describe, image)
