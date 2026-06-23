import json
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".85")

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
from keras import ops

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gemma4.configuration import load_config
from examples.gemma4.image_converter import preprocess_images
from examples.gemma4.inference import (Gemma4MultimodalDecoderStep,
                                       Gemma4PerLayerEmbeddingStep)
from examples.gemma4.multimodal_decoding import generate
from examples.gemma4.tokenizer import Gemma4Tokenizer
from examples.gemma4.vision import VisionEncoderArgs, build_vision_encoder

WEIGHTS = Path(__file__).resolve().with_name("weights")


def encode_image(path, config, vision_args, weights_dir):
    # Vision encoding (full attention over all patches) is memory-heavy, so run
    # it first and free it before loading the 8.7 GB of text weights.
    vision = build_vision_encoder(
        vision_args, weights_path=weights_dir / "vision_encoder.weights.h5")
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    image = (image / 255.0).astype("float32")[None]
    inputs = preprocess_images(image, vision_args)
    positions = np.array(inputs["pixel_position_ids"])[0]
    pool = vision_args.pool_size
    width = (int(positions[:, 0].max()) + 1) // pool
    height = (int(positions[:, 1].max()) + 1) // pool
    valid = width * height
    scale = float(config.hidden_dim) ** -0.5
    embeddings = np.array(vision(inputs))[0][:valid] * scale
    del vision
    gc.collect()
    return embeddings.astype("float32")


def build_text(config, weights_dir):
    # Load the large embedding table first in isolation, then the decoder step,
    # to keep peak memory low (mirrors demo_e2b.py).
    per_layer = Gemma4PerLayerEmbeddingStep(config)
    per_layer.load_weights(str(weights_dir / "embedding_step.weights.h5"))
    gc.collect()
    step = Gemma4MultimodalDecoderStep(config)
    step.load_weights(str(weights_dir / "decoder_step.weights.h5"))
    gc.collect()
    return step, per_layer


def build_prompt(tokenizer, num_image_tokens, question):
    head = tokenizer.tokenize("<|turn>user\n")[1:]
    tail = tokenizer.tokenize("{}<turn|>\n<|turn>model\n".format(question))[1:]
    image_block = ([tokenizer.start_of_image_token_id]
                   + [tokenizer.image_placeholder_id] * num_image_tokens
                   + [tokenizer.end_of_image_token_id])
    tokens = [2] + head + image_block + tail
    indices = [i for i, t in enumerate(tokens)
               if t == tokenizer.image_placeholder_id]
    return tokens, indices


def caption(image_path, question, max_tokens, weights_dir=WEIGHTS):
    config = load_config(weights_dir / "config.json")
    vision_args = VisionEncoderArgs(
        **json.load(open(weights_dir / "vision_config.json")))
    embeddings = encode_image(image_path, config, vision_args, weights_dir)
    step, per_layer = build_text(config, weights_dir)
    embeddings = ops.cast(embeddings, config.dtype)
    tokenizer = Gemma4Tokenizer(weights_dir / "tokenizer.json", add_bos=True)
    tokens, indices = build_prompt(tokenizer, len(embeddings), question)
    stop = tokenizer.get_stop_token_ids()[-1]
    generated = generate(step, per_layer, embeddings, config, tokens, indices,
                         stop, max_tokens)
    return tokenizer.detokenize(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma4 image captioning demo")
    add = parser.add_argument
    add("--image", required=True)
    add("--question", default="Describe this image.")
    add("--max_tokens", default=32, type=int)
    args = parser.parse_args()
    print(caption(args.image, args.question, args.max_tokens))
