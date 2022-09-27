import argparse
from paz.applications import SSD512MinimalHandPose
from paz.backend.camera import VideoPlayer, Camera


parser = argparse.ArgumentParser(description='''Minimal hand detection and
                                 keypoints estimation''')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = SSD512MinimalHandPose(right_hand=False, offsets=[0.5, 0.5])
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
