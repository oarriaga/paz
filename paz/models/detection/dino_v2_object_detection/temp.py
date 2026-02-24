import io
import os

import cv2
import numpy as np
from PIL import Image
from urllib.request import urlopen

from paz.models.detection.dino_v2_object_detection.utils.coco_classes import COCO_CLASSES
from paz.models.detection.dino_v2_object_detection.detr import (
    RFDETRBase as K_RFDETRBase,
    RFDETRNano as K_RFDETRNano,
    RFDETRSmall as K_RFDETRSmall,
    RFDETRMedium as K_RFDETRMedium,
    RFDETRLarge as K_RFDETRLarge,
)
from rfdetr import (
    RFDETRBase as PT_RFDETRBase,
    RFDETRNano as PT_RFDETRNano,
    RFDETRSmall as PT_RFDETRSmall,
    RFDETRMedium as PT_RFDETRMedium,
    RFDETRLarge as PT_RFDETRLarge,
)

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_visual_compare")
THRESHOLD = 0.3

COCO_IMAGES = {
    "cats": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "bear": "http://images.cocodataset.org/val2017/000000000285.jpg",
    "kitchen": "http://images.cocodataset.org/val2017/000000037777.jpg",
}

VARIANTS = {
    "nano": (K_RFDETRNano, PT_RFDETRNano),
    "small": (K_RFDETRSmall, PT_RFDETRSmall),
    "medium": (K_RFDETRMedium, PT_RFDETRMedium),
    "base": (K_RFDETRBase, PT_RFDETRBase),
    "large": (K_RFDETRLarge, PT_RFDETRLarge),
}

# ── Drawing helpers (pure OpenCV, no supervision) ────────────────────────────
COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
]


def _color_for(class_id):
    return COLORS[class_id % len(COLORS)]


def _draw_detections(image, boxes, scores, labels, title=""):
    """Draw boxes + labels on a copy of the image. Returns BGR."""
    vis = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    for box, score, cid in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = _color_for(int(cid))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        name = COCO_CLASSES.get(int(cid), str(int(cid)))
        txt = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis,
            txt,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    # Add title banner at top
    if title:
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return vis


def _load_image(url):
    data = urlopen(url).read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def _pad_height(im, target_h):
    if im.shape[0] < target_h:
        pad = np.zeros((target_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
        return np.vstack([im, pad])
    return im


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Download all images first
    print("Downloading test images...")
    images = {name: _load_image(url) for name, url in COCO_IMAGES.items()}
    print(f"  Loaded {len(images)} images\n")

    for var_name, (keras_cls, pt_cls) in VARIANTS.items():
        print(f"=== {var_name.upper()} ===")
        k_model = keras_cls()
        pt_model = pt_cls()

        for img_name, img in images.items():
            # ── Keras prediction ──
            k_res = k_model.predict(img, threshold=THRESHOLD)[0]
            k_vis = _draw_detections(
                img,
                k_res["boxes"],
                k_res["scores"],
                k_res["labels"],
                title=f"KERAS {var_name} ({len(k_res['scores'])} dets)",
            )
            k_path = os.path.join(OUT_DIR, f"{var_name}_{img_name}_keras.jpg")
            cv2.imwrite(k_path, k_vis)

            # ── PT prediction ──
            pt_det = pt_model.predict(img, threshold=THRESHOLD)
            pt_vis = _draw_detections(
                img,
                pt_det.xyxy,
                pt_det.confidence,
                pt_det.class_id,
                title=f"PT {var_name} ({len(pt_det.confidence)} dets)",
            )
            pt_path = os.path.join(OUT_DIR, f"{var_name}_{img_name}_pt.jpg")
            cv2.imwrite(pt_path, pt_vis)

            # ── Side-by-side comparison ──
            h = max(k_vis.shape[0], pt_vis.shape[0])
            side = np.hstack([_pad_height(k_vis, h), _pad_height(pt_vis, h)])
            side_path = os.path.join(OUT_DIR, f"{var_name}_{img_name}_compare.jpg")
            cv2.imwrite(side_path, side)

            k_n = len(k_res["scores"])
            p_n = len(pt_det.confidence)
            print(f"  {img_name}: keras={k_n} dets, pt={p_n} dets")

        del k_model, pt_model
        print()

    print(f"\nAll images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
