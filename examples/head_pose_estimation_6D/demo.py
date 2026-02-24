import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument("--camera", default=4, type=int)
parser.add_argument("--box_scale", default=1.2, type=float)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()


camera = paz.Camera(identifier=args.camera)
camera.intrinsics = camera.intrinsics_from_HFOV()
pipeline = paz.applications.HeadPoseKeypointNet2D32(camera, args.box_scale)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
