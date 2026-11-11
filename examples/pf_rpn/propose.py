"""Class-agnostic object proposals with PF-RPN.

Runs out of the box: with no arguments it downloads the pretrained weights
and three demo images of different sizes, then writes each image with its
proposals drawn on it. Every image is resized to the model's fixed input, so
the model compiles once however the image sizes vary.
"""
import argparse
import time
import urllib.request

import jax

import paz
from paz.applications.detectors import PFRPN, draw_proposals
from paz.models.detection.pf_rpn import download_weights

DEMO_URL = "http://images.cocodataset.org/val2017/"
DEMO_IMAGES = ["000000039769.jpg", "000000000785.jpg", "000000397133.jpg"]


def fetch_demo_images():
    for name in DEMO_IMAGES:
        urllib.request.urlretrieve(DEMO_URL + name, name)
    return DEMO_IMAGES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--IOU_thresh", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    model = paz.models.PFRPN((*args.size, 3))
    model.load_weights(download_weights(model))
    forward = jax.jit(lambda x: model(x))
    arguments = args.size, args.IOU_thresh, args.top_k, draw_proposals
    propose = PFRPN(forward, *arguments)
    paths = fetch_demo_images() if args.images is None else args.images
    for index, path in enumerate(paths):
        image = paz.image.load(path)
        start = time.perf_counter()
        (boxes, scores), drawn = propose(image)
        seconds = time.perf_counter() - start
        size = paz.image.get_size(image)
        print(f"{path}: {size} {len(boxes)} proposals in {seconds:.3f} s")
        paz.image.write(f"proposals_{index}.png", drawn)
    print("compilations:", forward._cache_size())
