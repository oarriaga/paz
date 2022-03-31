import argparse
import numpy as np

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.pipelines import HeadPoseKeypointNet2D32

description = 'Demo script for estimating 6D pose-heads from face-keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-fl', '--focal_length', type=float, default=None,
                    help="Focal length in pixels. If ''None'' it's"
                    "approximated using the image width")
parser.add_argument('-ic', '--image_center', nargs='+', type=float,
                    default=None, help="Image center in pixels for internal"
                    "camera matrix. If ''None'' it's approximated using the"
                    "image center from an extracted frame.")
args = parser.parse_args()

# obtaining a frame to perform focal-length and camera center approximation
camera = Camera(args.camera_id)
camera.start()
image_size = camera.read().shape[0:2]
camera.stop()

# loading focal length or approximating it
focal_length = args.focal_length
if focal_length is None:
    focal_length = image_size[1]

# loading image/sensor center or approximating it
image_center = args.image_center
if args.image_center is None:
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

# building camera parameters
camera.distortion = np.zeros((4, 1))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])

pipeline = HeadPoseKeypointNet2D32(camera)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
