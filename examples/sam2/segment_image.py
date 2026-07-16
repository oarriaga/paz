"""Segment an object with SAM 2.1 from one positive and one negative point.

Converted weights come from a directory produced by
``paz.models.foundation.sam2.pretrained.convert_checkpoint``; pass it with
``--weights``. The positive point marks the object, the negative point marks
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--positive", type=int, nargs=2, default=(340, 260))
    parser.add_argument("--negative", type=int, nargs=2, default=(50, 40))
    parser.add_argument("--output", default="sam2_mask.png")
    args = parser.parse_args()

    model = SAM21HieraSmall(weights=args.weights)
    image = demo.fetch_image() if args.image is None else \
        paz.image.load(args.image)

    features = predict.encode_image(model, image)
    points = [args.positive, args.negative]
    masks, scores, low_res = predict.predict(
        model, features, points=points, labels=[1, 0], multimask=True)

    best = int(np.argmax(np.array(scores)[0]))
    mask = np.array(masks)[0, best] > 0
    overlaid = overlay_masks(image, mask.astype("int32"),
                             [BACKGROUND, FOREGROUND])
    paz.image.write(args.output, overlaid)
    print("best mask", best, "score", float(np.array(scores)[0, best]))
    print("saved", args.output)
