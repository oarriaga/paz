"""Dataset adapter for RFDETR / LWDETR training — DeepFish."""
import os
import glob
from collections import defaultdict

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

DEEPFISH_CLASS_NAMES = ["Fish"]


def _build_class_to_id(class_names):
    return {name: idx for idx, name in enumerate(class_names)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DeepFishDataset:
    """Dataset that serves (image, target) for DeepFish training.

    Reads YOLO-format ``.txt`` annotations that sit alongside ``.jpg``
    images inside ``Deepfish/{video_id}/{train,valid}/`` sub-directories.

    Parameters
    ----------
    root : str or None
        Root directory of the extracted DeepFish dataset.
        If ``None``, defaults to ``~/.keras/paz/datasets/Deepfish``.
    resolution : int or None
        If given, resize all images to ``(resolution, resolution)``.
    subset : int or None
        If given, limit the dataset to the first *subset* images.
    """

    def __init__(
        self,
        root=None,
        resolution=None,
        subset=None,
    ):
        if root is None:
            root = os.path.expanduser("~/.keras/paz/datasets/Deepfish")
        self.root = root
        self.resolution = resolution
        self.class_names = list(DEEPFISH_CLASS_NAMES)
        self._class_to_id = _build_class_to_id(self.class_names)

        # -- discover images + annotations across all video sub-dirs -----
        # Structure: Deepfish/{video_id}/{train,valid}/*.{jpg,txt}
        image_paths = sorted(glob.glob(os.path.join(root, "*", "*", "*.jpg")))

        self._img_ids = []       # list[str]           – unique ID per image
        self._img_paths = {}     # img_id  -> abs path
        self._annotations = defaultdict(list)  # img_id  -> list[row-dict]

        for img_path in image_paths:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            self._img_ids.append(img_id)
            self._img_paths[img_id] = img_path

            # Matching annotation file
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        # YOLO format: class_id cx cy w h (normalised)
                        cls_id = int(float(parts[0]))
                        cx_n, cy_n, w_n, h_n = (
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        )
                        # We store normalised coords; will convert to
                        # absolute when needed (in _build_target and
                        # prepare_coco_dataset).  Use dummy 1×1 so that
                        # the absolute coords equal the normalised ones.
                        x_min_n = cx_n - w_n / 2.0
                        x_max_n = cx_n + w_n / 2.0
                        y_min_n = cy_n - h_n / 2.0
                        y_max_n = cy_n + h_n / 2.0
                        label_str = self.class_names[min(cls_id, len(self.class_names) - 1)]
                        self._annotations[img_id].append({
                            "label_l1": label_str,
                            "x_min_norm": x_min_n,
                            "x_max_norm": x_max_n,
                            "y_min_norm": y_min_n,
                            "y_max_norm": y_max_n,
                        })

        # -- subset --------------------------------------------------------
        if subset is not None:
            self._img_ids = self._img_ids[:min(subset, len(self._img_ids))]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._img_ids)

    def get_image_path(self, img_id):
        """Return the absolute path of the image file for *img_id*."""
        return self._img_paths.get(img_id)

    def __getitem__(self, idx):
        img_id = self._img_ids[idx]
        image = self._load_image(img_id)
        target = self._build_target(img_id)

        # Resize
        if self.resolution is not None:
            image = np.array(
                Image.fromarray(
                    (image * 255).astype(np.uint8)
                ).resize(
                    (self.resolution, self.resolution), Image.BILINEAR
                )
            ).astype(np.float32) / 255.0

        return image, target

    @property
    def num_classes(self):
        return len(self.class_names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, img_id):
        """Load and return an HWC float32 [0, 1] image."""
        path = self._img_paths[img_id]
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0

    def _build_target(self, img_id):
        """Convert annotations into target dict.

        Boxes are already in normalised cxcywh from the YOLO format.
        """
        rows = self._annotations.get(img_id, [])
        boxes = []
        labels = []
        for row in rows:
            x_min_n = row["x_min_norm"]
            x_max_n = row["x_max_norm"]
            y_min_n = row["y_min_norm"]
            y_max_n = row["y_max_norm"]
            label_str = row["label_l1"].strip()

            if label_str not in self._class_to_id:
                continue

            # Clamp
            x_min_n = max(0.0, min(x_min_n, 1.0))
            x_max_n = max(0.0, min(x_max_n, 1.0))
            y_min_n = max(0.0, min(y_min_n, 1.0))
            y_max_n = max(0.0, min(y_max_n, 1.0))

            if x_max_n <= x_min_n or y_max_n <= y_min_n:
                continue

            cx = (x_min_n + x_max_n) / 2.0
            cy = (y_min_n + y_max_n) / 2.0
            w = x_max_n - x_min_n
            h = y_max_n - y_min_n

            boxes.append([cx, cy, w, h])
            labels.append(self._class_to_id[label_str])

        if len(boxes) == 0:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)
        else:
            boxes_arr = np.array(boxes, dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64)

        return {"boxes": boxes_arr, "labels": labels_arr}
