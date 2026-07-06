import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz

parser = argparse.ArgumentParser(description="Minimal hand keypoint detection")
parser.add_argument("-c", "--camera_id", type=int, default=0)
parser.add_argument("--right_hand", action="store_true")
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()

estimate = paz.applications.MinimalHandPoseEstimation(right_hand=args.right_hand)
camera = paz.Camera(identifier=args.camera_id)
player = paz.VideoPlayer((args.H, args.W), estimate, camera)
player.run()
