import argparse

from paz.backend.camera import VideoPlayer, Camera
import paz.pipelines.detection as dt
from vvadlrs3 import pretrained_models

parser = argparse.ArgumentParser(description='Minimal hand detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

# model = pretrained_models.getFaceImageModel()

pipeline = dt.DetectVVAD()
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
# player = VideoPlayer((640, 640), pipeline, camera)
player.run()
