import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz

parser = argparse.ArgumentParser(description="EfficientDet-D0 COCO demo")
parser.add_argument("--image", default=None, help="RGB image; webcam if unset")
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--score_thresh", default=0.6, type=float)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--output", default="efficientdet.jpg")
args = parser.parse_args()

detect = paz.applications.EFFICIENTDETD0COCO(score_thresh=args.score_thresh)

if args.image is not None:
    inferences, image = detect(paz.image.load(args.image))
    paz.image.write(args.output, image)
else:
    camera = paz.Camera(identifier=args.camera)
    player = paz.VideoPlayer((args.H, args.W), detect, camera)
    player.run()
