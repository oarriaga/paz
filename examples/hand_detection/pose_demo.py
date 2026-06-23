import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz

parser = argparse.ArgumentParser(description="Hand detection + pose demo")
parser.add_argument("--image", default=None, help="RGB image; webcam if unset")
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--box_scale", default=1.5, type=float)
parser.add_argument("--right_hand", action="store_true")
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--output", default="hand_pose.jpg")
args = parser.parse_args()

estimate = paz.applications.DetectMinimalHand(
    box_scale=args.box_scale, right_hand=args.right_hand)

if args.image is not None:
    inferences, image = estimate(paz.image.load(args.image))
    paz.image.write(args.output, image)
else:
    camera = paz.Camera(identifier=args.camera)
    player = paz.VideoPlayer((args.H, args.W), estimate, camera)
    player.run()
