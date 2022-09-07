import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.applications import ClassifyHandClosure

parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = ClassifyHandClosure(draw=True, right_hand=False)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
