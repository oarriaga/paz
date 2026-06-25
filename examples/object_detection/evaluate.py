import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"

import paz
from paz.applications.detectors import SSD

def build_ssd300_voc():
    model = paz.models.detection.SSD300(21, "VOC", "VOC", (300, 300, 3))
    builder = paz.models.detection.single_shot_detector.build_prior_boxes
    return model, builder("VOC")


def build_efficientdet_d0_voc():
    model = paz.models.EFFICIENTDETD0(21, "VOC", "VOC")
    return model, model.prior_boxes


MODELS = {"SSD300VOC": build_ssd300_voc,
          "EFFICIENTDETD0VOC": build_efficientdet_d0_voc}

parser = argparse.ArgumentParser(description="Detector mean average precision")
parser.add_argument("--model", default="SSD300VOC", choices=list(MODELS))
parser.add_argument("--dataset", default="VOC2007", type=str)
parser.add_argument("--split", default="test", type=str)
parser.add_argument("--score_thresh", default=0.01, type=float)
parser.add_argument("--nms_thresh", default=0.45, type=float)
parser.add_argument("--iou_thresh", default=0.5, type=float)
parser.add_argument("--top_k", default=200, type=int)
parser.add_argument("--use_07_metric", action="store_true")
args = parser.parse_args()

model, prior_boxes = MODELS[args.model]()
class_names = paz.datasets.labels("VOC")
num_classes = len(class_names)
variances = [0.1, 0.1, 0.2, 0.2]
nms = paz.lock(paz.detection.apply_per_class_NMS, num_classes, args.nms_thresh,
               args.top_k)
detector = SSD(model, args.score_thresh, prior_boxes, variances, nms, None)

images, ground_truths = paz.datasets.load(args.dataset, args.split)
difficulties = paz.datasets.voc.load_difficulties(args.dataset, args.split)
result = paz.evaluation.compute_mAP(
    detector,
    images,
    ground_truths,
    num_classes,
    difficulties=difficulties,
    iou_thresh=args.iou_thresh,
    use_07_metric=args.use_07_metric,
    verbose=True,
)
for class_arg, class_name in enumerate(class_names):
    print(f"{class_name:>15}: {result['ap'][class_arg]:.4f}")
print(f"{'mAP':>15}: {result['mAP']:.4f}")
