import argparse
from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from pose import EFFICIENTPOSEALINEMOD

description = 'Demo script for object detection using EfficientDet model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

camera = Camera(args.camera_id)
pipeline = EFFICIENTPOSEALINEMOD(show_boxes2D=True, show_poses6D=True)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
