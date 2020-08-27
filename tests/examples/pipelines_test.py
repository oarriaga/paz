import argparse

import numpy as np

from paz.backend.image import load_image, show_image, resize_image

from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER
from paz.pipelines import DetectFaceKeypointNet2D32
from paz.pipelines import HeadPoseKeypointNet2D32
from paz.pipelines import SSD300FAT, SSD300VOC, SSD512COCO, SSD512YCBVideo

parser = argparse.ArgumentParser(description='Real-time face classifier')
parser.add_argument('-o', '--offset', type=float, default=0.1,
                    help='Scaled offset to be added to bounding boxes')
parser.add_argument('-s', '--score_thresh', type=float, default=0.6,
                    help='Box/class score threshold')
parser.add_argument('-n', '--nms_thresh', type=float, default=0.45,
                    help='non-maximum suppression threshold')
parser.add_argument('-p', '--image_path', type=str,
                    help='full image path used for the pipelines')
parser.add_argument('-c', '--camera_id', type=str,
                    help='Camera/device ID')
parser.add_argument('-d', '--dataset', type=str, default='COCO',
                    choices=['VOC', 'COCO', 'YCBVideo', 'FAT'],
                    help='Dataset name')
args = parser.parse_args()



name_to_model = {'VOC': SSD300VOC, 'FAT': SSD300FAT, 'COCO': SSD512COCO,
                 'YCBVideo': SSD512YCBVideo}


image = load_image(args.image_path)
H = 1000
W = int((H / image.shape[0]) * image.shape[1])
# image = resize_image(image, (W, H))

focal_length = image.shape[1]
image_center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
camera = Camera(args.camera_id)
camera.distortion = np.zeros((4, 1))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])

pipeline_A = DetectMiniXceptionFER([args.offset, args.offset])
pipeline_B = DetectFaceKeypointNet2D32()
pipeline_C = HeadPoseKeypointNet2D32(camera)
pipeline_D = name_to_model[args.dataset](args.score_thresh, args.nms_thresh)
pipelines = [pipeline_A, pipeline_B, pipeline_C, pipeline_D]
for pipeline in pipelines:
    predictions = pipeline(image.copy())
    show_image(predictions['image'])
