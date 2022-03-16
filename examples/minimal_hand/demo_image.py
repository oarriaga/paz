import argparse
import os
from paz.backend.image import load_image, write_image
from pipelines import MANOHandPoseEstimation

description = 'Demo script for estimating 6D pose-heads from face-keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-i', '--images_path', type=str,
                    default='images',
                    help='Directory for the test images')
args = parser.parse_args()

image = load_image(os.path.join(args.images_path, 'hand3.jpg'))
detect = MANOHandPoseEstimation()
inferences = detect(image)

image = inferences['image']
write_image('./output/result.jpg', image)
