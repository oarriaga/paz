import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()


pipeline = paz.applications.DetectFaceKeypointNet2D32()
camera = paz.Camera(identifier=args.camera)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
