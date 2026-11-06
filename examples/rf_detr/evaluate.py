"""Scores a fine-tuned RF-DETR on VOC2007 and draws what it found.

Reports mean average precision at IOU 0.5, at 0.75, and averaged over the
ten COCO thresholds, which punishes loose boxes the 0.5 number lets through.
One pass over the images feeds all three. --num_drawn writes that many
images with the ground truth beside the detections.
"""
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

import paz

VARIANTS = {
    "nano": paz.models.TrainableRFDETRNano,
    "small": paz.models.TrainableRFDETRSmall,
    "medium": paz.models.TrainableRFDETRMedium,
    "base": paz.models.TrainableRFDETRBase,
    "large": paz.models.TrainableRFDETRLarge,
}


def build_detector(variant, num_classes, num_groups, weights, score_thresh):
    """Rebuilds the graph that trained, so a grouped checkpoint also loads.

    Keras keys a weights file by layer class and position, so the model that
    reads a checkpoint has to be built the same way as the one that wrote it.
    """
    model = VARIANTS[variant](num_classes, num_groups)
    model.load_weights(weights)
    detector = paz.models.detection.rf_detr.build_detector(model)
    return paz.applications.detectors.RFDETR(detector, score_thresh, None)


def draw_comparison(image, ground_truth, detections, names, colors):
    """Ground truth on the left, detections on the right."""
    truth = np.asarray(ground_truth, "float32")
    scores = np.ones(len(truth), "float32")
    boxes = truth[:, :4].astype("int32")
    args = boxes, truth[:, 4].astype("int32"), scores
    expected = paz.draw.boxes2D(image, *args, names, colors)
    predicted = paz.draw.boxes2D(image, *detections, names, colors)
    return np.concatenate([expected, predicted], axis=1)


def write_comparisons(detector, paths, ground_truths, names, root, thresh):
    """One image per path: ground truth beside the confident detections."""
    colors = paz.draw.lincolor(len(names))
    for index, path in enumerate(paths):
        image = paz.image.load(path)
        detections = keep_confident(detector(image), thresh)
        args = image, ground_truths[index], detections, names, colors
        drawn = draw_comparison(*args)
        paz.image.write(os.path.join(root, f"detections_{index}.png"), drawn)


def keep_confident(detections, thresh):
    """Average precision wants every box, a drawing only the sure ones."""
    boxes, labels, scores = detections
    chosen = np.asarray(scores) >= thresh
    return boxes[chosen], labels[chosen], scores[chosen]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", default="nano", choices=list(VARIANTS))
    parser.add_argument("--dataset", default="VOC2007")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_groups", default=1, type=int)
    parser.add_argument("--score_thresh", default=0.05, type=float)
    parser.add_argument("--num_images", default=None, type=int)
    parser.add_argument("--num_drawn", default=0, type=int)
    parser.add_argument("--draw_thresh", default=0.5, type=float)
    parser.add_argument("--root", default="evaluation")
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)

    names = paz.datasets.voc.get_class_names()
    detector_args = (args.variant, len(names), args.num_groups,
                     args.weights)
    detector = build_detector(*detector_args, args.score_thresh)
    images, ground_truths = paz.datasets.voc.load(args.dataset, args.split)
    difficulties = paz.datasets.voc.load_difficulties(args.dataset, args.split)
    if args.num_images is not None:
        images = images[:args.num_images]
        ground_truths = ground_truths[:args.num_images]
        difficulties = difficulties[:args.num_images]

    scored = detector, images, ground_truths, len(names)
    kwargs = dict(difficulties=difficulties, verbose=True)
    result = paz.evaluation.compute_COCO_mAP(*scored, **kwargs)
    for class_arg, class_name in enumerate(names):
        print(f"{class_name:>15}: {result['ap'][class_arg]:.4f}")
    print(f"{'mAP@0.5':>15}: {result['mAP_50']:.4f}")
    print(f"{'mAP@0.75':>15}: {result['mAP_75']:.4f}")
    print(f"{'mAP@[.5:.95]':>15}: {result['mAP']:.4f}")
    if args.num_drawn > 0:
        drawn = args.num_drawn
        drawn_args = detector, images[:drawn], ground_truths[:drawn], names
        write_comparisons(*drawn_args, args.root, args.draw_thresh)
        print("wrote", drawn, "comparisons to", args.root)
