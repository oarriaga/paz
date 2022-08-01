import argparse
from paz.applications import DetectMinimalHand
from paz.applications import MinimalHandPoseEstimation
from paz.pipelines.detection import SSD512HandDetection
from paz.backend.camera import VideoPlayer, Camera


parser = argparse.ArgumentParser(description='Minimal hand detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = DetectMinimalHand(
    SSD512HandDetection(), MinimalHandPoseEstimation(right_hand=False))
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
