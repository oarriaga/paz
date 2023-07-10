import argparse
from paz.applications import PIX2YCBTools6D
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera


parser = argparse.ArgumentParser(description='Object pose estimation')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

camera = Camera(args.camera_id)
pipeline = PIX2YCBTools6D(camera, offsets=[0.25, 0.25], epsilon=0.015)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
