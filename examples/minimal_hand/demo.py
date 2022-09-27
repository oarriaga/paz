import argparse
from paz.applications import MinimalHandPoseEstimation
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = MinimalHandPoseEstimation(right_hand=False)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
