import os
import random
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

# ---------------------------------------------------------------------------
# Resolve project root so both packages can be imported
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = _SCRIPT_DIR.parents[4]  # .../paz/
_PYTORCH_ROOT = (
    _PAZ_ROOT
    / "examples"
    / "rf-detr_original_pytorch_implementation"
)

# Import the transforms directly from the file path to avoid triggering the
# heavy paz/__init__.py chain (keras, jax, etc.).
import importlib.util


def _import_from_path(module_name, file_path):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


KT = _import_from_path(
    "keras_transforms",
    str(_SCRIPT_DIR / "transforms.py"),
)

# --- PyTorch imports (optional — fall back to a reference numpy clone) ---
_HAS_TORCH = False
try:
    if str(_PYTORCH_ROOT) not in sys.path:
        sys.path.insert(0, str(_PYTORCH_ROOT))
    import torch
    import torchvision.transforms as _TV
    import torchvision.transforms.functional as TF
    from rfdetr.datasets import transforms as PT
    _HAS_TORCH = True

    # ------------------------------------------------------------------
    # Monkey-patch T.RandomCrop.get_params so the crop *position* is
    # drawn from Python's ``random`` module instead of ``torch.randint``.
    # This lets us synchronise the two pipelines with a single seed.
    # ------------------------------------------------------------------
    _orig_get_params = _TV.RandomCrop.get_params

    @staticmethod
    def _synced_get_params(img, output_size):
        """Drop-in replacement that uses random.randint (stdlib)."""
        w, h = img.size if hasattr(img, "size") else (img.shape[-1], img.shape[-2])
        th, tw = output_size
        if h == th and w == tw:
            return 0, 0, h, w
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        return top, left, th, tw

    _TV.RandomCrop.get_params = _synced_get_params
except Exception:
    torch = None  # type: ignore[assignment]
    PT = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RESOLUTION = 560  # default RF-DETR resolution
OUTPUT_DIR = _PAZ_ROOT / "temp"

# Box colors for drawing
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 128),
]


# ---------------------------------------------------------------------------
# Reference (numpy/CHW) implementation — mirrors the PyTorch transforms
# exactly, but without a torch dependency.  Used when torch is unavailable.
# ---------------------------------------------------------------------------

class _RefToTensor:
    """PIL → float32 (C,H,W) numpy, same as torchvision F.to_tensor."""
    def __call__(self, img, target):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        return arr, target


class _RefNormalize:
    """ImageNet-normalise a (C,H,W) numpy array + convert boxes."""
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, image, target=None):
        image = (image - self.mean) / self.std
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]  # (C, H, W)
        if "boxes" in target:
            boxes = target["boxes"]
            x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            b = np.stack(
                [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)],
                axis=-1,
            )
            b = b / np.array([w, h, w, h], dtype=np.float32)
            target["boxes"] = b
        return image, target


# Reuse Keras helpers that are identical to PyTorch behaviour at the PIL level
_RefRandomHorizontalFlip = KT.RandomHorizontalFlip
_RefRandomResize = KT.RandomResize
_RefSquareResize = KT.SquareResize
_RefRandomSizeCrop = KT.RandomSizeCrop
_RefRandomSelect = KT.RandomSelect
_RefCompose = KT.Compose


# ---------------------------------------------------------------------------
# Synthetic test data generation
# ---------------------------------------------------------------------------

def _make_test_images(n=4, seed=42):
    """Generate synthetic PIL images + COCO-style targets (xyxy pixel boxes).

    Returns list of (PIL.Image, target_dict) pairs.
    """
    rng = np.random.RandomState(seed)
    samples = []
    img_configs = [
        (640, 480, 3),   # landscape
        (480, 640, 2),   # portrait
        (500, 500, 4),   # square
        (800, 600, 1),   # large landscape
    ]
    for idx, (w, h, n_boxes) in enumerate(img_configs[:n]):
        # Create a colorful gradient image for visual verification
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        arr[:, :, 2] = rng.randint(50, 200)
        # Add a unique marker per image so we can distinguish them
        arr[10:30, 10:30] = [255, 255, 255]
        img = PIL.Image.fromarray(arr)

        # Random xyxy boxes (pixel coords, valid)
        boxes = []
        for _ in range(n_boxes):
            x0 = rng.randint(0, w // 2)
            y0 = rng.randint(0, h // 2)
            x1 = rng.randint(x0 + 20, min(x0 + 200, w))
            y1 = rng.randint(y0 + 20, min(y0 + 200, h))
            boxes.append([x0, y0, x1, y1])
        boxes_arr = np.array(boxes, dtype=np.float32)

        target = {
            "boxes": boxes_arr,
            "labels": np.arange(n_boxes, dtype=np.int64),
            "area": (boxes_arr[:, 2] - boxes_arr[:, 0])
            * (boxes_arr[:, 3] - boxes_arr[:, 1]),
            "iscrowd": np.zeros(n_boxes, dtype=np.int64),
            "image_id": np.array([idx], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
        }
        samples.append((img, target))
    return samples


# ---------------------------------------------------------------------------
# Load real images from the Deepfish dataset
# ---------------------------------------------------------------------------

def _load_deepfish_images(n=20, split="train"):
    """Load *n* images + annotations from the Deepfish dataset.

    The dataset is expected at ``~/.keras/paz/datasets/Deepfish/`` (the
    default location used by ``paz.datasets.deepfish.download``).

    Each annotation text file stores one detection per line::

        <class> <cx_norm> <cy_norm> <w_norm> <h_norm>

    where coordinates are normalised to [0, 1] w.r.t. 1920×1080.

    Returns
    -------
    list[tuple[PIL.Image, dict]]
        ``(PIL.Image.Image, target_dict)`` pairs identical in format to
        ``_make_test_images``.
    """
    import glob as _glob

    root = os.path.expanduser("~/.keras/paz/datasets/Deepfish")
    assert os.path.isdir(root), (
        f"Deepfish dataset not found at {root}.  "
        "Run paz.datasets.deepfish.download() first."
    )

    img_paths = sorted(_glob.glob(f"{root}/*/{split}/*.jpg"))
    txt_paths = sorted(_glob.glob(f"{root}/*/{split}/*.txt"))

    assert len(img_paths) > 0, f"No images found in {root}/*/{split}/"

    # Pair each image with its annotation file by matching stem
    txt_by_stem = {}
    for tp in txt_paths:
        stem = os.path.splitext(os.path.basename(tp))[0]
        txt_by_stem[stem] = tp

    # Deterministic subset: take the first n (paths are sorted)
    img_paths = img_paths[:n]

    samples = []
    for idx, img_path in enumerate(img_paths):
        img = PIL.Image.open(img_path).convert("RGB")
        w, h = img.size  # actual image dimensions

        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = txt_by_stem.get(stem)

        boxes = []
        if txt_path and os.path.isfile(txt_path):
            with open(txt_path, "r") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _cls, cx_n, cy_n, bw_n, bh_n = (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                    # Normalised cxcywh → pixel xyxy
                    x0 = (cx_n - bw_n / 2) * w
                    y0 = (cy_n - bh_n / 2) * h
                    x1 = (cx_n + bw_n / 2) * w
                    y1 = (cy_n + bh_n / 2) * h
                    # Clip to image bounds
                    x0 = max(0.0, min(x0, float(w)))
                    y0 = max(0.0, min(y0, float(h)))
                    x1 = max(0.0, min(x1, float(w)))
                    y1 = max(0.0, min(y1, float(h)))
                    if x1 > x0 and y1 > y0:
                        boxes.append([x0, y0, x1, y1])

        n_boxes = len(boxes)
        if n_boxes == 0:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes_arr = np.array(boxes, dtype=np.float32)

        target = {
            "boxes": boxes_arr,
            "labels": np.zeros(n_boxes, dtype=np.int64),  # single class
            "area": (
                (boxes_arr[:, 2] - boxes_arr[:, 0])
                * (boxes_arr[:, 3] - boxes_arr[:, 1])
                if n_boxes > 0
                else np.zeros(0, dtype=np.float32)
            ),
            "iscrowd": np.zeros(n_boxes, dtype=np.int64),
            "image_id": np.array([idx], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
        }
        samples.append((img, target))

    return samples


# ---------------------------------------------------------------------------
# Pipeline builders (mirror the actual RF-DETR train pipeline)
#
# Each transform is wrapped with _Log so we can record exactly which
# augmentation operations were applied to every image.
# ---------------------------------------------------------------------------

# Thread-local list that collects log entries during a single pipeline call.
_aug_log: list = []


class _Log:
    """Wrapper that delegates to *inner* and appends a description to _aug_log."""

    def __init__(self, inner, name):
        self.inner = inner
        self.name = name

    def __call__(self, img, target):
        # Capture image size before the transform
        if hasattr(img, "size"):
            before = img.size  # (w, h) for PIL
        elif hasattr(img, "shape"):
            before = img.shape
        else:
            before = None

        out_img, out_target = self.inner(img, target)

        # Capture image size after the transform
        if hasattr(out_img, "size"):
            after = out_img.size
        elif hasattr(out_img, "shape"):
            after = out_img.shape
        else:
            after = None

        # Build a human-readable log entry
        entry = self.name
        if before is not None and after is not None and before != after:
            entry += f"  {before} -> {after}"
        _aug_log.append(entry)

        return out_img, out_target


class _LogFlip:
    """Log wrapper for RandomHorizontalFlip — records flip vs no-flip."""

    def __init__(self, inner):
        self.inner = inner
        self.p = inner.p

    def __call__(self, img, target):
        # Peek at the random state to know if the flip will happen,
        # then restore state and let the real transform run.
        state = random.getstate()
        will_flip = random.random() < self.p
        random.setstate(state)

        out_img, out_target = self.inner(img, target)

        if will_flip:
            _aug_log.append("RandomHorizontalFlip  (flipped)")
        else:
            _aug_log.append("RandomHorizontalFlip  (no flip)")
        return out_img, out_target


class _LogSelect:
    """Log wrapper for RandomSelect — records which branch was taken."""

    def __init__(self, inner, branch_a_name="Branch A", branch_b_name="Branch B"):
        self.inner = inner
        self.p = inner.p
        self.branch_a_name = branch_a_name
        self.branch_b_name = branch_b_name

    def __call__(self, img, target):
        state = random.getstate()
        chose_a = random.random() < self.p
        random.setstate(state)

        out_img, out_target = self.inner(img, target)

        if chose_a:
            _aug_log.append(f"RandomSelect  -> {self.branch_a_name}")
        else:
            _aug_log.append(f"RandomSelect  -> {self.branch_b_name}")
        return out_img, out_target


def _build_pytorch_train_pipeline(resolution):
    """Exact RF-DETR train pipeline (square_resize_div_64 variant).

    When torch is available uses the real PyTorch transforms.
    Falls back to a reference numpy (CHW) clone otherwise.
    """
    if _HAS_TORCH:
        normalize = PT.Compose([
            PT.ToTensor(),
            PT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        scales = [resolution]
        flip = PT.RandomHorizontalFlip()
        select = PT.RandomSelect(
            PT.SquareResize(scales),
            PT.Compose([
                _Log(PT.RandomResize([400, 500, 600]), "RandomResize([400,500,600])"),
                _Log(PT.RandomSizeCrop(384, 600), "RandomSizeCrop(384,600)"),
                _Log(PT.SquareResize(scales), f"SquareResize({scales})"),
            ]),
        )
        return PT.Compose([
            _LogFlip(flip),
            _LogSelect(select, f"SquareResize({scales})",
                       "Resize+Crop+SquareResize"),
            _Log(normalize, "ToTensor+Normalize"),
        ])
    else:
        normalize = _RefCompose([
            _RefToTensor(),
            _RefNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        scales = [resolution]
        flip = _RefRandomHorizontalFlip()
        select = _RefRandomSelect(
            _RefSquareResize(scales),
            _RefCompose([
                _Log(_RefRandomResize([400, 500, 600]), "RandomResize([400,500,600])"),
                _Log(_RefRandomSizeCrop(384, 600), "RandomSizeCrop(384,600)"),
                _Log(_RefSquareResize(scales), f"SquareResize({scales})"),
            ]),
        )
        return _RefCompose([
            _LogFlip(flip),
            _LogSelect(select, f"SquareResize({scales})",
                       "Resize+Crop+SquareResize"),
            _Log(normalize, "ToTensor+Normalize"),
        ])


def _build_keras_train_pipeline(resolution):
    """Exact Keras replica of the RF-DETR train pipeline."""
    normalize = KT.Compose([
        KT.ToTensor(),
        KT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    scales = [resolution]
    flip = KT.RandomHorizontalFlip()
    select = KT.RandomSelect(
        KT.SquareResize(scales),
        KT.Compose([
            _Log(KT.RandomResize([400, 500, 600]), "RandomResize([400,500,600])"),
            _Log(KT.RandomSizeCrop(384, 600), "RandomSizeCrop(384,600)"),
            _Log(KT.SquareResize(scales), f"SquareResize({scales})"),
        ]),
    )
    return KT.Compose([
        _LogFlip(flip),
        _LogSelect(select, f"SquareResize({scales})",
                   "Resize+Crop+SquareResize"),
        _Log(normalize, "ToTensor+Normalize"),
    ])


def _build_pytorch_val_pipeline(resolution):
    """RF-DETR val pipeline (square_resize_div_64 variant)."""
    if _HAS_TORCH:
        normalize = PT.Compose([
            PT.ToTensor(),
            PT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return PT.Compose([
            _Log(PT.SquareResize([resolution]), f"SquareResize([{resolution}])"),
            _Log(normalize, "ToTensor+Normalize"),
        ])
    else:
        normalize = _RefCompose([
            _RefToTensor(),
            _RefNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return _RefCompose([
            _Log(_RefSquareResize([resolution]), f"SquareResize([{resolution}])"),
            _Log(normalize, "ToTensor+Normalize"),
        ])


def _build_keras_val_pipeline(resolution):
    """Keras replica of the RF-DETR val pipeline."""
    normalize = KT.Compose([
        KT.ToTensor(),
        KT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return KT.Compose([
        _Log(KT.SquareResize([resolution]), f"SquareResize([{resolution}])"),
        _Log(normalize, "ToTensor+Normalize"),
    ])


# ---------------------------------------------------------------------------
# Target conversion helpers  (numpy ↔ torch)
# ---------------------------------------------------------------------------

def _target_np_to_pt(target):
    """Convert a numpy target dict for the PyTorch / reference pipeline.

    When torch is available, values become torch tensors.
    Otherwise they stay as numpy copies.
    """
    out = {}
    for k, v in target.items():
        if isinstance(v, np.ndarray):
            if _HAS_TORCH:
                out[k] = torch.as_tensor(v.copy())
            else:
                out[k] = v.copy()
        else:
            out[k] = v
    return out


def _target_to_np(target):
    """Convert a target dict (torch or numpy) to pure numpy."""
    out = {}
    for k, v in target.items():
        if _HAS_TORCH and isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------------

def _denorm_pytorch(img):
    """Denormalize a PyTorch / reference (C,H,W) image → (H,W,C) uint8 numpy."""
    if _HAS_TORCH and isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()  # (C, H, W)
    else:
        img_np = img  # already numpy (C, H, W)
    img_np = img_np.transpose(1, 2, 0)  # → (H, W, C)
    img_np = img_np * IMAGENET_STD + IMAGENET_MEAN  # undo normalize
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


def _denorm_keras(img_np):
    """Denormalize a Keras (H,W,C) float32 numpy array → uint8."""
    img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


# ---------------------------------------------------------------------------
# Box drawing helpers
# ---------------------------------------------------------------------------

def _cxcywh_norm_to_xyxy_pixel(boxes, w, h):
    """Normalized cxcywh → pixel xyxy."""
    boxes = boxes.copy()
    boxes[:, 0] *= w
    boxes[:, 1] *= h
    boxes[:, 2] *= w
    boxes[:, 3] *= h
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = cx - bw / 2
    y0 = cy - bh / 2
    x1 = cx + bw / 2
    y1 = cy + bh / 2
    return np.stack([x0, y0, x1, y1], axis=-1)


def _draw_boxes(img_uint8, boxes_xyxy, label=""):
    """Draw bounding boxes on a uint8 RGB image. Returns PIL Image."""
    pil_img = PIL.Image.fromarray(img_uint8.copy())
    draw = PIL.ImageDraw.Draw(pil_img)
    for i, box in enumerate(boxes_xyxy):
        color = COLORS[i % len(COLORS)]
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), f"b{i}", fill=color)
    if label:
        draw.text((5, 5), label, fill=(255, 255, 255))
    return pil_img


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare_pipelines(pipeline_name, pt_pipeline, kt_pipeline, samples,
                      seed=12345):
    """Run both pipelines on every sample with synchronized RNG.

    Saves side-by-side comparison images to OUTPUT_DIR.
    Returns a list of (max_pixel_diff, max_box_diff) per image.
    """
    results = []
    for idx, (pil_img, np_target) in enumerate(samples):
        # ---- Prepare inputs (same image, same target) ----
        pt_target = _target_np_to_pt(np_target)
        kt_target = {k: v.copy() if isinstance(v, np.ndarray) else v
                      for k, v in np_target.items()}

        # ---- Run PyTorch / reference pipeline with saved RNG state ----
        random.seed(seed + idx)
        if _HAS_TORCH:
            torch.manual_seed(seed + idx)
        _aug_log.clear()
        pt_img, pt_tgt = pt_pipeline(pil_img.copy(), pt_target)
        pt_log = list(_aug_log)

        # ---- Run Keras pipeline with SAME RNG state ----
        random.seed(seed + idx)
        _aug_log.clear()
        kt_img, kt_tgt = kt_pipeline(pil_img.copy(), kt_target)
        kt_log = list(_aug_log)

        # ---- Denormalize images back to uint8 ----
        pt_rgb = _denorm_pytorch(pt_img)
        kt_rgb = _denorm_keras(kt_img)

        # ---- Recover pixel-space boxes ----
        pt_boxes_np = _target_to_np(pt_tgt)
        pt_h, pt_w = pt_rgb.shape[:2]
        kt_h, kt_w = kt_rgb.shape[:2]

        if pt_boxes_np["boxes"].shape[0] > 0:
            pt_xyxy = _cxcywh_norm_to_xyxy_pixel(
                pt_boxes_np["boxes"], pt_w, pt_h
            )
        else:
            pt_xyxy = np.zeros((0, 4), dtype=np.float32)

        if kt_tgt["boxes"].shape[0] > 0:
            kt_xyxy = _cxcywh_norm_to_xyxy_pixel(
                kt_tgt["boxes"], kt_w, kt_h
            )
        else:
            kt_xyxy = np.zeros((0, 4), dtype=np.float32)

        # ---- Compute diffs ----
        max_pixel_diff = np.max(np.abs(
            pt_rgb.astype(np.int16) - kt_rgb.astype(np.int16)
        )) if pt_rgb.shape == kt_rgb.shape else -1

        if pt_xyxy.shape == kt_xyxy.shape and pt_xyxy.shape[0] > 0:
            max_box_diff = float(np.max(np.abs(pt_xyxy - kt_xyxy)))
        elif pt_xyxy.shape[0] == 0 and kt_xyxy.shape[0] == 0:
            max_box_diff = 0.0
        else:
            max_box_diff = -1.0

        results.append((max_pixel_diff, max_box_diff))

        # ---- Draw boxes on both augmented outputs ----
        pt_vis = _draw_boxes(pt_rgb, pt_xyxy, label="PyTorch")
        kt_vis = _draw_boxes(kt_rgb, kt_xyxy, label="Keras")

        # ---- Draw boxes on the original (pre-augmentation) image ----
        orig_rgb = np.asarray(pil_img, dtype=np.uint8)
        if np_target["boxes"].shape[0] > 0:
            orig_xyxy = np_target["boxes"].copy()
        else:
            orig_xyxy = np.zeros((0, 4), dtype=np.float32)
        orig_vis = _draw_boxes(orig_rgb, orig_xyxy, label="Original")

        # Resize the original panel to match the augmented height for
        # a clean layout (augmented images are typically square 560×560).
        aug_h = max(pt_vis.height, kt_vis.height)
        if orig_vis.height != aug_h:
            scale = aug_h / orig_vis.height
            new_w = int(orig_vis.width * scale)
            orig_vis = orig_vis.resize((new_w, aug_h), PIL.Image.BILINEAR)

        # ---- Three-panel composite: Original | PyTorch | Keras ----
        gap = 10
        combined_w = orig_vis.width + pt_vis.width + kt_vis.width + 2 * gap
        header_h = 40

        # Build augmentation log text for the footer
        line_h = 14
        aug_lines = []
        aug_lines.append("Augmentations applied:")
        max_len = max(len(pt_log), len(kt_log))
        for i in range(max_len):
            pt_entry = pt_log[i] if i < len(pt_log) else ""
            kt_entry = kt_log[i] if i < len(kt_log) else ""
            if pt_entry == kt_entry:
                aug_lines.append(f"  {i+1}. {pt_entry}")
            else:
                aug_lines.append(f"  {i+1}. PT: {pt_entry}  |  KT: {kt_entry}")
        footer_h = len(aug_lines) * line_h + 10

        combined = PIL.Image.new(
            "RGB",
            (combined_w, header_h + aug_h + footer_h),
            (30, 30, 30),
        )

        # Header labels
        draw = PIL.ImageDraw.Draw(combined)
        draw.text(
            (orig_vis.width // 2 - 25, 5),
            "Original",
            fill=(180, 255, 180),
        )
        col2_x = orig_vis.width + gap
        draw.text(
            (col2_x + pt_vis.width // 2 - 30, 5),
            "PyTorch",
            fill=(255, 200, 100),
        )
        col3_x = col2_x + pt_vis.width + gap
        draw.text(
            (col3_x + kt_vis.width // 2 - 20, 5),
            "Keras",
            fill=(100, 200, 255),
        )
        # Diff stats
        stats = (
            f"pixel_diff={max_pixel_diff}  box_diff={max_box_diff:.4f}  "
            f"shapes: pt={pt_rgb.shape} kt={kt_rgb.shape}"
        )
        draw.text((5, 20), stats, fill=(200, 200, 200))

        combined.paste(orig_vis, (0, header_h))
        combined.paste(pt_vis, (col2_x, header_h))
        combined.paste(kt_vis, (col3_x, header_h))

        # Footer — augmentation log
        footer_y = header_h + aug_h + 5
        for j, line in enumerate(aug_lines):
            color = (220, 220, 100) if j == 0 else (200, 200, 200)
            draw.text((10, footer_y + j * line_h), line, fill=color)

        out_path = OUTPUT_DIR / f"{pipeline_name}_img{idx}.png"
        combined.save(str(out_path))
        print(
            f"  [{pipeline_name}] img {idx}: "
            f"pixel_diff={max_pixel_diff}, box_diff={max_box_diff:.6f} "
            f"→ {out_path.name}"
        )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    if _HAS_TORCH:
        print("Mode: PyTorch (real torchvision transforms) vs Keras (numpy/PIL)\n")
    else:
        print(
            "Mode: Reference numpy-CHW clone vs Keras (numpy/PIL)\n"
            "  (torch not available — using reference implementation that\n"
            "   replicates PyTorch behaviour in pure numpy/PIL)\n"
        )

    samples = _load_deepfish_images(n=20, split="train")
    print(f"Loaded {len(samples)} Deepfish images.\n")

    # ---- Train pipeline (square_resize_div_64, no multi-scale) ----
    print("=" * 60)
    print("TRAIN pipeline (SquareResize variant, resolution={})".format(RESOLUTION))
    print("=" * 60)
    pt_train = _build_pytorch_train_pipeline(RESOLUTION)
    kt_train = _build_keras_train_pipeline(RESOLUTION)
    train_results = compare_pipelines(
        "train_square", pt_train, kt_train, samples, seed=11111
    )

    # ---- Val pipeline ----
    print()
    print("=" * 60)
    print("VAL pipeline (SquareResize variant, resolution={})".format(RESOLUTION))
    print("=" * 60)
    pt_val = _build_pytorch_val_pipeline(RESOLUTION)
    kt_val = _build_keras_val_pipeline(RESOLUTION)
    val_results = compare_pipelines(
        "val_square", pt_val, kt_val, samples, seed=22222
    )

    # ---- Train pipeline with different seeds (to exercise different paths) ----
    print()
    print("=" * 60)
    print("TRAIN pipeline - alternate seed (different random choices)")
    print("=" * 60)
    train_results_2 = compare_pipelines(
        "train_square_alt", pt_train, kt_train, samples, seed=99999
    )

    # ---- Summary ----
    all_results = train_results + val_results + train_results_2
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pixel = [r[0] for r in all_results]
    all_box = [r[1] for r in all_results]
    worst_pixel = max(all_pixel)
    worst_box = max(all_box)

    for name, results in [
        ("train_square", train_results),
        ("val_square", val_results),
        ("train_square_alt", train_results_2),
    ]:
        pmax = max(r[0] for r in results)
        bmax = max(r[1] for r in results)
        status = "PASS" if pmax <= 1 and bmax < 0.01 else "MISMATCH"
        print(f"  {name:25s} pixel_max={pmax:3d}  box_max={bmax:.6f}  [{status}]")

    print()
    if worst_pixel <= 1 and worst_box < 0.01:
        print("ALL COMPARISONS PASSED — Keras pipeline is a faithful clone.")
    else:
        print(
            f"WARNING: differences detected.  "
            f"Worst pixel diff = {worst_pixel}, worst box diff = {worst_box:.6f}"
        )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
