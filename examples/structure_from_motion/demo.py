import argparse
import os
from paz.backend.image import load_image, show_image
from paz.backend.camera import Camera
from pipeline import ComputeHomography, FeatureDetector, StructureFromMotion


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-i', '--images_path', type=str, default='images',
                    help='Directory for images')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=75, help='Horizontal field of view in degrees')
args = parser.parse_args()


camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
camera_inrinsics = camera.intrinsics

images = []
for filename in os.listdir(args.images_path):
    images.append(load_image(os.path.join(args.images_path, filename)))


# detect = FeatureDetector()
# detect = ComputeHomography()
detect = StructureFromMotion(camera_inrinsics)
inferences = detect(images)
# show_image(inferences['keypoint_image'])
# show_image(inferences['match_image'])
# show_image(inferences['image'])
