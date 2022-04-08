import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from pipelines import MANOHandPoseEstimation


parser = argparse.ArgumentParser(description='Test keypoints network')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = MANOHandPoseEstimation()
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
