import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz

parser = argparse.ArgumentParser(description="HigherHRNet 2D human pose demo")
parser.add_argument("--image", default=None, help="RGB image; webcam if unset")
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--output", default="human_pose_2D.jpg")
args = parser.parse_args()

estimate = paz.applications.HigherHRNetHumanPose2D()

if args.image is not None:
    inferences, image = estimate(paz.image.load(args.image))
    paz.image.write(args.output, image)
else:
    camera = paz.Camera(identifier=args.camera)
    player = paz.VideoPlayer((args.H, args.W), estimate, camera)
    player.run()
