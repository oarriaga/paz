import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"
import paz

WEIGHTS = "experiments/imdb_mini_XCEPTION_paz_jax.weights.h5"

parser = argparse.ArgumentParser(description="MiniXception IMDB demo")
parser.add_argument("--weights", default=WEIGHTS)
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()


names = paz.datasets.labels("IMDB")
model = paz.models.build_mini_xception_imdb((64, 64, 1), len(names))
model.load_weights(args.weights)
classify = paz.applications.ClassifyMiniXception(model)
pipeline = paz.applications.DetectMiniXception(classify, names, 1.2, None)
camera = paz.Camera(args.camera)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
