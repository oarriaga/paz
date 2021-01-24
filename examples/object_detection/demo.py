import argparse

from paz.pipelines import SSD300FAT, SSD300VOC, SSD512COCO, SSD512YCBVideo
from paz.backend.camera import VideoPlayer, Camera
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)




parser = argparse.ArgumentParser(description='SSD object detection demo')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-s', '--score_thresh', type=float, default=0.6,
                    help='Box/class score threshold')
parser.add_argument('-n', '--nms_thresh', type=float, default=0.45,
                    help='non-maximum suppression threshold')
parser.add_argument('-d', '--dataset', type=str, default='VOC',
                    choices=['VOC', 'COCO', 'YCBVideo', 'FAT'],
                    help='Dataset name')
args = parser.parse_args()
name_to_model = {'VOC': SSD300VOC,
                 'FAT': SSD300FAT,
                 'COCO': SSD512COCO,
                 'YCBVideo': SSD512YCBVideo}

pipeline = name_to_model[args.dataset]
detect = pipeline(args.score_thresh, args.nms_thresh)
camera = Camera(args.camera_id)
player = VideoPlayer((1280, 960), detect, camera)
player.run()
