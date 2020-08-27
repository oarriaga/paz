import argparse
from pipelines import DetectGMMKeypointNet2D
from paz.backend.image import show_image, load_image

description = 'Demo for visualizing uncertainty in probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', type=str, help='Path to image')
args = parser.parse_args()

pipeline = DetectGMMKeypointNet2D()
image = load_image(args.path)
inferences = pipeline(image)
show_image(inferences['image'])
show_image(inferences['contours'][0])
