"""Segment an object with SAM 2.1 from one positive and one negative point.

Runs out of the box: downloads the pretrained SAM 2.1 small weights and a demo
image, then overlays the highest-quality mask. The application jits the image
encoder, so it fits in a few gigabytes of RAM.
"""
import argparse

import paz

import demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--positive", type=int, nargs=2, default=(340, 260))
    parser.add_argument("--negative", type=int, nargs=2, default=(50, 40))
    parser.add_argument("--output", default="sam2_mask.png")
    args = parser.parse_args()

    segment = paz.applications.SAMHieraSmall21()
    image = paz.image.load(args.image) if args.image else demo.fetch_image()
    points = [args.positive, args.negative]
    masks, scores, overlay = segment(image, points=points, labels=[1, 0])
    paz.image.write(args.output, overlay)
    print("saved", args.output)
