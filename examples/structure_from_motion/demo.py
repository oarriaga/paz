import argparse
import os
from paz.backend.image import load_image
from pipeline_cv2 import StructureFromMotion
# from pipeline_np import StructureFromMotion
import numpy as np
from backend import camera_intrinsics_from_dfov


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
root = os.path.expanduser('~/DFKI/paz/examples/structure_from_motion/datasets')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-i', '--images_path', type=str,
                    default='datasets/SixDPose/cheezIt_textured',
                    # default='datasets/SixDPose/cheezIt',
                    # default='datasets/images1',
                    help='Directory for images')
parser.add_argument('-DFOV', '--diagonal_field_of_view', type=float,
                    default=54, help='Diagonal field of view in degrees')
args = parser.parse_args()


camera_intrinsics = np.array([[568.996140852, 0, 643.21055941],
                              [0, 568.988362396, 477.982801038],
                              [0, 0, 1]])

images = []

# image_files = os.listdir(args.images_path)
# for filename in image_files:
#     image = load_image(os.path.join(args.images_path, filename))
#     images.append(image)

# detect = StructureFromMotion(camera_intrinsics)
# inferences = detect(images)


# for custom objects
image_files = os.listdir(args.images_path)
image_files = sorted(image_files, key=lambda f: int(f.split('.')[0]))
for filename in image_files:
    image = load_image(os.path.join(args.images_path, filename))
    images.append(image)

H, W = images[0].shape[:2]
camera_intrinsics = camera_intrinsics_from_dfov(
    args.diagonal_field_of_view, H, W)

f = np.sqrt(H ** 2 + W ** 2)
camera_intrinsics = np.asarray([[f, 0, W/2],
                                [0, f, H/2],
                                [0, 0, 1]], np.float32)

print(camera_intrinsics)
detect = StructureFromMotion(camera_intrinsics)
# inferences = detect(images[30:40])
inferences = detect(images[:8])
