"""Segment an object with SAM 2.1 from one positive and one negative point.

Runs out of the box: it downloads the pretrained SAM 2.1 small weights and a
demo image. The positive point marks the object, the negative point marks
background; the mask with the highest predicted quality is overlaid.
"""
import argparse

import numpy as np

import paz
from paz.models import SAM21HieraSmall
from paz.models.foundation.sam2 import predict
from paz.backend.draw import overlay_masks

import demo

BACKGROUND = (0, 0, 0)
FOREGROUND = (30, 144, 255)


def load_image(path):
    return demo.fetch_image() if path is None else paz.image.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="pretrained")
    parser.add_argument("--image", default=None)
    parser.add_argument("--positive", type=int, nargs=2, default=(340, 260))
    parser.add_argument("--negative", type=int, nargs=2, default=(50, 40))
    parser.add_argument("--output", default="sam2_mask.png")
    args = parser.parse_args()

    model = SAM21HieraSmall(weights=args.weights)
    image = load_image(args.image)
    state = predict.encode_image(model, image)
    prompt = dict(points=[args.positive, args.negative], labels=[1, 0])
    masks, scores, _ = predict.predict(state, **prompt)
    masks, scores = predict.select(masks, scores, multimask=True)

    best = int(np.argmax(np.array(scores)[0]))
    mask = np.array(masks)[0, best] > 0
    colors = [BACKGROUND, FOREGROUND]
    overlaid = overlay_masks(image, mask.astype("int32"), colors)
    paz.image.write(args.output, overlaid)
    print("best mask", best, "score", float(np.array(scores)[0, best]))
    print("saved", args.output)
