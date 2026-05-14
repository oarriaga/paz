from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Union, Optional

try:
    import supervision as sv
except ImportError:
    sv = None

def _xywh_to_xyxy(boxes):
    """Convert ``[x, y, w, h]`` boxes to ``[x1, y1, x2, y2]`` format.

    Args:
        boxes (list[list[float]]): List of boxes in ``[x, y, w, h]``.

    Returns:
        np.ndarray: ``(N, 4)`` array in ``[x1, y1, x2, y2]`` format.
    """
    if not boxes:
        return np.empty((0, 4))
    arr = np.array(boxes)
    xyxy = np.zeros_like(arr)
    xyxy[:, 0] = arr[:, 0]
    xyxy[:, 1] = arr[:, 1]
    xyxy[:, 2] = arr[:, 0] + arr[:, 2]
    xyxy[:, 3] = arr[:, 1] + arr[:, 3]
    return xyxy


def save_gt_predictions_visualization(
    scenario_name: str,
    image_width: int,
    image_height: int,
    gt_boxes: List[List[float]],
    gt_class_ids: List[int],
    pred_boxes: List[List[float]],
    pred_class_ids: List[int],
    pred_confidences: List[float],
    pred_ious: List[Optional[float]],
    save_dir: Path,
) -> None:
    """Save a visualization image with ground-truth and predicted boxes.

    Creates a blank canvas with a top padding region, draws GT boxes
    in green/cyan and prediction boxes in red/magenta, labels each
    with class ID and (for predictions) confidence and optional IoU,
    then saves the result as a PNG.

    Args:
        scenario_name (str): Base filename for the saved image.
        image_width (int): Canvas width in pixels.
        image_height (int): Canvas height in pixels.
        gt_boxes (list): Ground-truth boxes as ``[x, y, w, h]``.
        gt_class_ids (list[int]): Class IDs for each GT box.
        pred_boxes (list): Predicted boxes as ``[x, y, w, h]``.
        pred_class_ids (list[int]): Class IDs for each prediction.
        pred_confidences (list[float]): Confidence scores.
        pred_ious (list[float | None]): IoU values (``None`` if unavailable).
        save_dir (Path): Output directory for the saved image.
    """
    if sv is None:
        print("Supervision library not found. Skipping visualization.")
        return

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)

    # Create a blank canvas with top padding for header space
    top_padding = 60
    image = np.zeros((image_height + top_padding, image_width, 3), dtype=np.uint8)

    # Offset boxes vertically to account for the padding region
    gt_boxes_offset = [[x, y + top_padding, w, h] for x, y, w, h in gt_boxes]
    pred_boxes_offset = [[x, y + top_padding, w, h] for x, y, w, h in pred_boxes]

    gt_xyxy = _xywh_to_xyxy(gt_boxes_offset)
    pred_xyxy = _xywh_to_xyxy(pred_boxes_offset)

    gt_detections = None
    pred_detections = None

    if len(gt_xyxy) > 0:
        gt_detections = sv.Detections(
            xyxy=gt_xyxy,
            class_id=np.array(gt_class_ids),
        )

    if len(pred_xyxy) > 0:
        pred_detections = sv.Detections(
            xyxy=pred_xyxy,
            class_id=np.array(pred_class_ids),
            confidence=np.array(pred_confidences),
        )

    # Color palettes: index 0 is unused (class IDs are 1-indexed)
    gt_colors = sv.ColorPalette.from_hex(['#808080', '#00ff64', '#00c8ff'])
    pred_colors = sv.ColorPalette.from_hex(['#808080', '#ff6432', '#ff32c8'])

    gt_box_annotator = sv.BoxAnnotator(
        color=gt_colors, thickness=3, color_lookup=sv.ColorLookup.CLASS
    )
    pred_box_annotator = sv.BoxAnnotator(
        color=pred_colors, thickness=3, color_lookup=sv.ColorLookup.CLASS
    )

    gt_label_annotator = sv.LabelAnnotator(
        color=gt_colors,
        text_color=sv.Color.BLACK,
        text_scale=0.5,
        text_padding=3,
        text_position=sv.Position.TOP_LEFT,
        color_lookup=sv.ColorLookup.CLASS,
    )
    pred_label_annotator = sv.LabelAnnotator(
        color=pred_colors,
        text_color=sv.Color.BLACK,
        text_scale=0.5,
        text_padding=3,
        text_position=sv.Position.TOP_RIGHT,
        color_lookup=sv.ColorLookup.CLASS,
    )

    gt_labels = [f"c{class_id}" for class_id in gt_class_ids]

    pred_labels = []
    for class_id, conf, iou in zip(pred_class_ids, pred_confidences, pred_ious):
        if iou is not None:
            pred_labels.append(f"c{class_id}\nconf={conf:.3f}\niou={iou:.3f}")
        else:
            pred_labels.append(f"c{class_id}\nconf={conf:.3f}")

    if gt_detections is not None:
        image = gt_box_annotator.annotate(scene=image, detections=gt_detections)
        image = gt_label_annotator.annotate(
            scene=image, detections=gt_detections, labels=gt_labels
        )
    if pred_detections is not None:
        image = pred_box_annotator.annotate(scene=image, detections=pred_detections)
        image = pred_label_annotator.annotate(
            scene=image, detections=pred_detections, labels=pred_labels
        )

    Image.fromarray(image).save(save_dir_path / f"{scenario_name}.png")
    print(f"Saved visualization to {save_dir}/{scenario_name}.png")
