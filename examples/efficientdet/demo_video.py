import argparse
from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from detection import EFFICIENTDETD0COCO

description = 'Demo script for object detection using EfficientDet model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=70, help='Horizontal field of view in degrees')
args = parser.parse_args()

camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
pipeline = EFFICIENTDETD0COCO()
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
