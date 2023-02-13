import argparse
import os
from paz.backend.image import load_image, show_image
from pipeline1 import StructureFromMotion
import numpy as np


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-i', '--images_path', type=str,
                    # default='datasets/dtu/scan1/images',
                    default='images1',
                    help='Directory for images')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=75, help='Horizontal field of view in degrees')
args = parser.parse_args()


camera_intrinsics = np.array([[568.996140852, 0, 643.21055941],
                              [0, 568.988362396, 477.982801038],
                              [0, 0, 1]])


# camera_intrinsics = np.array([[2892.33, 0, 823.205],
#                               [0, 2883.18, 619.071],
#                               [0, 0, 1]])

images = []
for filename in os.listdir(args.images_path):
    images.append(load_image(os.path.join(args.images_path, filename)))


detect = StructureFromMotion(camera_intrinsics)
inferences = detect(images)
# show_image(inferences['keypoint_image'])
# show_image(inferences['match_image'])
# show_image(inferences['image'])
