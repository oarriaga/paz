from pathlib import Path
import numpy as np
from PIL import Image

try:
    import supervision as sv
except ImportError:
    sv = None

TOP_PADDING = 60
GROUND_TRUTH_COLORS = ['#808080', '#00ff64', '#00c8ff']
PREDICTION_COLORS = ['#808080', '#ff6432', '#ff32c8']


def xywh_to_xyxy(boxes):
    if not boxes:
        corners = np.empty((0, 4))
    else:
        boxes = np.array(boxes)
        corners = np.zeros_like(boxes)
        corners[:, 0] = boxes[:, 0]
        corners[:, 1] = boxes[:, 1]
        corners[:, 2] = boxes[:, 0] + boxes[:, 2]
        corners[:, 3] = boxes[:, 1] + boxes[:, 3]
    return corners


def offset_boxes(boxes, offset):
    return [[x, y + offset, w, h] for x, y, w, h in boxes]


def build_detections(boxes, class_ids, confidences=None):
    corners = xywh_to_xyxy(offset_boxes(boxes, TOP_PADDING))
    detections = None
    if len(corners) > 0:
        keys = ("xyxy", "class_id")
        values = (corners, np.array(class_ids))
        if confidences is not None:
            keys = keys + ("confidence",)
            values = values + (np.array(confidences),)
        detections = sv.Detections(**dict(zip(keys, values)))
    return detections


def build_box_annotator(palette):
    keys = ("color", "thickness", "color_lookup")
    values = (palette, 3, sv.ColorLookup.CLASS)
    return sv.BoxAnnotator(**dict(zip(keys, values)))


def build_label_annotator(palette, position):
    keys = ("color", "text_color", "text_scale", "text_padding", "text_position", "color_lookup")  # fmt: skip
    values = (palette, sv.Color.BLACK, 0.5, 3, position, sv.ColorLookup.CLASS)
    return sv.LabelAnnotator(**dict(zip(keys, values)))


def build_prediction_labels(class_ids, confidences, overlaps):
    labels = []
    for class_id, confidence, overlap in zip(class_ids, confidences, overlaps):
        label = f"c{class_id}\nconf={confidence:.3f}"
        if overlap is not None:
            label = label + f"\niou={overlap:.3f}"
        labels.append(label)
    return labels


def annotate_detections(image, detections, palette, labels, position):
    if detections is not None:
        box_annotator = build_box_annotator(palette)
        image = box_annotator.annotate(scene=image, detections=detections)
        label_annotator = build_label_annotator(palette, position)
        kwargs = dict(scene=image, detections=detections, labels=labels)
        image = label_annotator.annotate(**kwargs)
    return image


def compose_comparison_figure(width, height, ground_truth, predictions):
    # A blank canvas with top padding leaves room for the label header.
    image = np.zeros((height + TOP_PADDING, width, 3), dtype=np.uint8)
    # Index 0 of each palette is unused: class IDs are 1-indexed.
    palette = sv.ColorPalette.from_hex(GROUND_TRUTH_COLORS)
    args = (palette, ground_truth[1], sv.Position.TOP_LEFT)
    image = annotate_detections(image, ground_truth[0], *args)
    palette = sv.ColorPalette.from_hex(PREDICTION_COLORS)
    args = (palette, predictions[1], sv.Position.TOP_RIGHT)
    return annotate_detections(image, predictions[0], *args)


def save_gt_predictions_visualization(scenario_name, image_width, image_height, gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_confidences, pred_ious, save_dir):  # fmt: skip
    if sv is None:
        print("Supervision library not found. Skipping visualization.")
    else:
        directory = Path(save_dir)
        directory.mkdir(exist_ok=True, parents=True)
        labels = [f"c{class_id}" for class_id in gt_class_ids]
        ground_truth = (build_detections(gt_boxes, gt_class_ids), labels)
        args = (pred_boxes, pred_class_ids, pred_confidences)
        detections = build_detections(*args)
        labels = build_prediction_labels(pred_class_ids, pred_confidences, pred_ious)  # fmt: skip
        args = (image_width, image_height, ground_truth, (detections, labels))
        image = compose_comparison_figure(*args)
        Image.fromarray(image).save(directory / f"{scenario_name}.png")
        print(f"Saved visualization to {save_dir}/{scenario_name}.png")
