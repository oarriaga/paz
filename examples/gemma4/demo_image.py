import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

from keras.utils import get_file

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

import paz
from paz.applications import DescribeImageGemma4
from paz.models import Gemma4
from paz.models.foundation.gemma4.pretrained import (
    resolve_gemma4_dir, load_gemma4_vision_encoder)

SAMPLE_IMAGE_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.9.1/image_with_everyday_classes.jpg"  # fmt: skip


def download_sample_image():
    name = os.path.basename(SAMPLE_IMAGE_URL)
    path = get_file(name, SAMPLE_IMAGE_URL, cache_subdir="paz/images")
    return paz.image.load(path)


def build_models_with_cpu_vision(model_name, model_dir):
    # The bf16 language model runs on the default device (GPU). The fp32 vision
    # encoder (float32 by design, to match the reference) does not fit beside it
    # on a 16 GB GPU, so build that piece on the CPU and swap it in.
    models = Gemma4(model_name, models_path=model_dir)
    with jax.default_device(jax.devices("cpu")[0]):
        vision_encoder = load_gemma4_vision_encoder(model_dir)
    return models._replace(vision_encoder=vision_encoder)


def chat_about_image(describe, image):
    # Encode the image once; every question then reuses these embeddings and
    # only runs the fast, streaming text decode.
    embeddings = describe.encode(image)
    print("Ask about the image. Submit an empty line to quit.")
    while True:
        question = input("you> ").strip()
        if question == "":
            break
        print("gemma> ", end="", flush=True)
        describe.answer(embeddings, question)


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
    model_dir = resolve_gemma4_dir(args.model_name, args.models_path)
    models = build_models_with_cpu_vision(args.model_name, model_dir)
    describe = DescribeImageGemma4(
        args.model_name, args.max_tokens, models_path=model_dir, models=models)
    if args.image is None:
        image = download_sample_image()
    else:
        image = paz.image.load(args.image)
    chat_about_image(describe, image)
