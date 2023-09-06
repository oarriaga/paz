import argparse
from scipy.optimize import least_squares
from paz.applications import EstimateHumanPose
from paz.backend.camera import Camera, VideoPlayer


parser = argparse.ArgumentParser(description='Estimate human pose')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(HFOV=70, image_shape=(640, 480))
pipeline = EstimateHumanPose(least_squares, camera.intrinsics)
camera = Camera()
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
