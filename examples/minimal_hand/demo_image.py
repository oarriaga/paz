import argparse
import os
from paz.backend.image import load_image, show_image
from pipelines import MANOHandPoseEstimation

description = 'Demo script for estimating 6D pose-heads from face-keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-i', '--images_path', type=str,
                    default='images',
                    help='Directory for the test images')
args = parser.parse_args()

image = load_image(os.path.join(args.images_path, 'hand2.jpg'))
detect = MANOHandPoseEstimation()
inferences = detect(image)

image = inferences['image']
show_image(image)
