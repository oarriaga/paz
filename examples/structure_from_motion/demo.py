import argparse
import os
import matplotlib
matplotlib.use('WXAgg')
from matplotlib import pyplot as plt
print("Switched to:", matplotlib.get_backend())
import matplotlib.pyplot as plt

from paz.backend.image import load_image, show_image
<<<<<<< HEAD
from pipeline3 import StructureFromMotion
=======
from pipeline import StructureFromMotion
>>>>>>> 18713980afda1d2e4624397634a5f4dbeab958d1
import numpy as np


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
root = os.path.expanduser('~')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-i', '--images_path', type=str,
<<<<<<< HEAD
                    default='datasets/dtu/scan1/images',
                    # default='images1',
=======
                    default=os.path.join(root, 'dtu/scan1/images'),
>>>>>>> 18713980afda1d2e4624397634a5f4dbeab958d1
                    help='Directory for images')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=75, help='Horizontal field of view in degrees')
args = parser.parse_args()


# camera_intrinsics = np.array([[568.996140852, 0, 643.21055941],
#                               [0, 568.988362396, 477.982801038],
#                               [0, 0, 1]])

<<<<<<< HEAD

camera_intrinsics = np.array([[2892.33, 0, 823.205],
                              [0, 2883.18, 619.071],
                              [0, 0, 1]])

=======
camera_intrinsics = np.array([[2892.33, 0, 823.205],
                              [0, 2883.18, 619.071],
                              [0, 0, 1]])
>>>>>>> 18713980afda1d2e4624397634a5f4dbeab958d1

images = []
for filename in os.listdir(args.images_path):
    image = load_image(os.path.join(args.images_path, filename))
    # show_image(image)
    images.append(load_image(os.path.join(args.images_path, filename)))


detect = StructureFromMotion(camera_intrinsics)
inferences = detect(images)
# show_image(inferences['keypoint_image'])
# show_image(inferences['match_image'])
# show_image(inferences['image'])
