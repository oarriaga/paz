import argparse

from paz.backend.camera import VideoPlayer, Camera
from paz.models import SSD300, SSD512
from paz.datasets import get_class_names
from paz.pipelines import SingleShotInference


parser = argparse.ArgumentParser(description='MultiHaarCascadeDetectors')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-s', '--score_thresh', type=float, default=0.6,
                    help='Box/class score threshold')
parser.add_argument('-n', '--nms_thresh', type=float, default=0.45,
                    help='non-maximum suppression threshold')
parser.add_argument('-d', '--dataset', type=str, default='VOC',
                    choices=['VOC', 'COCO'], help='Dataset name')
args = parser.parse_args()

names = get_class_names(args.dataset)
if args.dataset == 'VOC':
    model = SSD300(len(names))
elif args.dataset == 'COCO':
    model = SSD512(len(names))
detect = SingleShotInference(model, names, args.score_thresh, args.nms_thresh)
camera = Camera(args.camera_id)
player = VideoPlayer((1280, 960), detect, camera)
player.run()
