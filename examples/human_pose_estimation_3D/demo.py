import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from scipy.optimize import least_squares

import paz

parser = argparse.ArgumentParser(description="Human pose 3D + 6D webcam demo")
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--HFOV", default=70, type=int)
args = parser.parse_args()

camera = paz.Camera(identifier=args.camera)
intrinsics = camera.intrinsics_from_HFOV(args.HFOV, (args.H, args.W))
pipeline = paz.applications.EstimateHumanPose(least_squares, intrinsics)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
