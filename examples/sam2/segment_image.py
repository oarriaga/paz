"""Interactively segment multiple objects in an image with SAM 2.1.

The image is encoded once with a jitted encoder, then each prompt runs the
light mask decoder. The encode-once/predict-many state lives here (developer
code) on the foundation ``predict`` module, so the applications API stays a
single high-level call.
"""
import os
import argparse
from collections import namedtuple

import numpy as np
import jax
from keras.utils import get_file

import paz
from paz.models.foundation.sam2 import predict
from paz.models.foundation.sam2.preprocessing import preprocess_image
from paz.models.foundation.sam2.prompt_encoder import dense_positional_encoding
from prompt_selector import PromptSelector

MAX_NUM_WARMUP_POINTS = 5
Segmenter = namedtuple("Segmenter", "encode_image predict")


def fetch_image():
    URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    path = get_file("sam2_cats.jpg", URL, cache_subdir="paz/examples/sam2")
    return paz.image.load(path)


def enable_compilation_cache():
    path = os.path.expanduser("~/.cache/paz/sam2_jax_cache")
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def build_segmenter(bundle):
    encode = jax.jit(lambda pixels: bundle.image_encoder(pixels))
    image_pe = dense_positional_encoding(bundle.point_encoder)[None]

    def encode_image(image):
        pixels = preprocess_image(image)[None]
        embedding, high_res_0, high_res_1 = encode(pixels)
        parts = embedding, high_res_0, high_res_1, image_pe, image.shape[:2]
        return predict.State(bundle, predict.Features(*parts))

    def predict_masks(state, points=None, labels=None, box=None):
        masks, scores, _ = predict.predict(state, points, labels, box)
        masks, scores = predict.select(masks, scores)
        return np.array(masks[0]) > 0, np.array(scores[0])

    return Segmenter(encode_image, predict_masks)


def segment_classes(segment, state, image, class_prompts):
    class_map = np.zeros(image.shape[:2], np.int32)
    colors = [(0, 0, 0)]
    for class_prompt in class_prompts:
        args = class_prompt.points, class_prompt.labels, class_prompt.box
        masks, scores = segment.predict(state, *args)
        best_mask = masks[int(np.argmax(scores))]
        class_map[best_mask] = len(colors)
        colors.append(class_prompt.color)
    return paz.draw.overlay_masks(image, class_map, colors)


def warmup_decoder(segment, state):
    for num_points in range(1, MAX_NUM_WARMUP_POINTS + 1):
        points = [(0, 0)] * num_points
        labels = [1] * num_points
        segment.predict(state, points, labels)
    segment.predict(state, box=(0, 0, 1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default="sam2_mask.png")
    args = parser.parse_args()

    enable_compilation_cache()
    image = paz.image.load(args.image) if args.image else fetch_image()
    segment = build_segmenter(paz.models.SAMHieraSmall21())
    state = segment.encode_image(image)
    print("Warming SAM prompt decoder...")
    warmup_decoder(segment, state)
    segment_prompts = paz.partial(segment_classes, segment, state, image)
    selector = PromptSelector(image, segment_prompts)
    result = selector.run()
    paz.image.write(args.output, result)
    print("saved", args.output)
