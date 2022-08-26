import argparse

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.pipelines import DetectFaceKeypointNet2D32

description = 'Demo script for running 2D keypoints face detector'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

# instantiating model
pipeline = DetectFaceKeypointNet2D32(radius=5)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
