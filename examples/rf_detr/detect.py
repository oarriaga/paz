"""COCO object detection with RF-DETR.

Runs out of the box: with no arguments it downloads the pretrained nano
weights and a demo image, then writes the image with the boxes drawn on it.
"""
import argparse
import urllib.request

import numpy as np

import paz
from paz.applications import DetectRFDETRNano, DetectRFDETRSmall
from paz.applications import DetectRFDETRMedium, DetectRFDETRBase
from paz.applications import DetectRFDETRLarge

DEMO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DETECTORS = {
    "nano": DetectRFDETRNano,
    "small": DetectRFDETRSmall,
    "medium": DetectRFDETRMedium,
    "base": DetectRFDETRBase,
    "large": DetectRFDETRLarge,
}


def fetch_demo_image(path="rf_detr_demo.jpg"):
    urllib.request.urlretrieve(DEMO_URL, path)
    return paz.image.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nano", choices=list(DETECTORS))
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default="detections.png")
    args = parser.parse_args()

    detect = DETECTORS[args.variant](args.score_thresh)
    if args.image is None:
        image = fetch_demo_image()
    else:
        image = paz.image.load(args.image)
    predictions, drawn = detect(image)

    names = ["0"] + paz.datasets.labels("COCO_EFFICIENTDET")
    for box, label, score in zip(*predictions):
        print(f"{names[label]:<16} {score:.3f} {np.array(box).tolist()}")
    paz.image.write(args.output, drawn)
    print("saved", args.output)
