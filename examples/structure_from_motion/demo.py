import argparse
import os
from paz.backend.image import load_image, show_image
from pipeline import ComputeHomography, FeatureDetector


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-i', '--images_path', type=str, default='images',
                    help='Directory for images')
args = parser.parse_args()

images = []
for filename in os.listdir(args.images_path):
    images.append(load_image(os.path.join(args.images_path, filename)))


# detect = FeatureDetector()
detect = ComputeHomography()
inferences = detect(images)
# show_image(inferences['keypoint_image'])
# show_image(inferences['match_image'])
show_image(inferences['image'])
