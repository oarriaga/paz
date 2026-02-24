import gc
import io
import os
import sys
import warnings

import numpy as np
import pytest
from urllib.request import urlopen

# ---- path setup ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- PyTorch guard -------------------------------------------------------
try:
    import torch
    from PIL import Image

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---- PyTorch RFDETR imports (detection only) -----------------------------
if HAS_TORCH:
    try:
        from rfdetr import (
            RFDETRBase as PT_RFDETRBase,
            RFDETRNano as PT_RFDETRNano,
            RFDETRSmall as PT_RFDETRSmall,
            RFDETRMedium as PT_RFDETRMedium,
            RFDETRLarge as PT_RFDETRLarge,
        )
    except ImportError:
        rfdetr_path = os.path.abspath(
            os.path.join(current_dir, "../../../../examples/rf-detr_original_pytorch_implementation")
        )
        if rfdetr_path not in sys.path:
            sys.path.insert(0, rfdetr_path)
        from rfdetr import (
            RFDETRBase as PT_RFDETRBase,
            RFDETRNano as PT_RFDETRNano,
            RFDETRSmall as PT_RFDETRSmall,
            RFDETRMedium as PT_RFDETRMedium,
            RFDETRLarge as PT_RFDETRLarge,
        )

    # XLarge / 2XLarge live under rfdetr.platform.models
    try:
        from rfdetr import (
            RFDETRXLarge as PT_RFDETRXLarge,
            RFDETR2XLarge as PT_RFDETR2XLarge,
        )
    except (ImportError, NameError):
        try:
            from rfdetr.platform.models import (
                RFDETRXLarge as PT_RFDETRXLarge,
                RFDETR2XLarge as PT_RFDETR2XLarge,
            )
        except (ImportError, NameError):
            PT_RFDETRXLarge = None
            PT_RFDETR2XLarge = None

    from rfdetr.util.misc import NestedTensor

# ---- Keras RFDETR imports (detection only) --------------------------------
from paz.models.detection.dino_v2_object_detection.detr import (
    RFDETRBase as K_RFDETRBase,
    RFDETRNano as K_RFDETRNano,
    RFDETRSmall as K_RFDETRSmall,
    RFDETRMedium as K_RFDETRMedium,
    RFDETRLarge as K_RFDETRLarge,
    RFDETRXLarge as K_RFDETRXLarge,
    RFDETR2XLarge as K_RFDETR2XLarge,
)
from paz.models.detection.dino_v2_object_detection.main import (
    PostProcess as K_PostProcess,
)
from paz.models.detection.dino_v2_object_detection.utils.coco_classes import COCO_CLASSES

# Weight-transfer utilities
if HAS_TORCH:
    from paz.models.detection.dino_v2_object_detection.models.lwdetr.test_lwdetr_with_real_weights import (
        transfer_full_model_weights,
        MODEL_CONFIGS,
    )

import keras
from keras import ops

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS_DIR = os.path.join(project_root, "rfdetr_keras_weights")
CACHE_DIR = os.path.join(project_root, ".test_cache")

COCO_IMAGES = {
    "cats": {
        "id": "000000039769",
        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "description": "Two cats on a couch with remotes",
        "expected_classes": {17},  # cat
    },
    "bear": {
        "id": "000000000285",
        "url": "http://images.cocodataset.org/val2017/000000000285.jpg",
        "description": "Bear in natural habitat",
        "expected_classes": {23},  # bear
    },
    "kitchen": {
        "id": "000000037777",
        "url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "description": "Kitchen scene with appliances and furniture",
        "expected_classes": {82},  # refrigerator
    },
}

# Detection-only variants (skip segmentation)
DETECTION_VARIANTS = {
    "RFDETRNano": {"keras_cls": K_RFDETRNano, "save_key": "rfdetr_nano"},
    "RFDETRSmall": {"keras_cls": K_RFDETRSmall, "save_key": "rfdetr_small"},
    "RFDETRMedium": {"keras_cls": K_RFDETRMedium, "save_key": "rfdetr_medium"},
    "RFDETRBase": {"keras_cls": K_RFDETRBase, "save_key": "rfdetr_base"},
    "RFDETRLarge": {"keras_cls": K_RFDETRLarge, "save_key": "rfdetr_large"},
    "RFDETRXLarge": {"keras_cls": K_RFDETRXLarge, "save_key": "rfdetr_xlarge"},
    "RFDETR2XLarge": {"keras_cls": K_RFDETR2XLarge, "save_key": "rfdetr_2xlarge"},
}

IMAGENET_MEANS = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STDS = np.array([0.229, 0.224, 0.225], dtype="float32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _download_coco_image(image_id, url):
    """Download or load cached COCO image. Returns (H, W, 3) uint8 RGB."""
    _ensure_cache_dir()
    cached = os.path.join(CACHE_DIR, f"coco_val_{image_id}.npy")
    if os.path.exists(cached):
        return np.load(cached)
    print(f"  Downloading COCO image {image_id} ...")
    data = urlopen(url).read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    np.save(cached, arr)
    return arr


def _preprocess(image_float, resolution):
    """Normalise and resize a float32 [0,1] image to (1, res, res, 3)."""
    normed = (image_float - IMAGENET_MEANS) / IMAGENET_STDS
    t = ops.convert_to_tensor(normed[np.newaxis], dtype="float32")
    resized = ops.image.resize(t, (resolution, resolution))
    return ops.convert_to_numpy(resized)


def _print_detections(scores, labels, header="", threshold=0.3):
    """Print detections above *threshold*."""
    keep = scores > threshold
    s = scores[keep]
    l = labels[keep]
    order = np.argsort(-s)
    prefix = f"  [{header}]" if header else "  "
    print(f"{prefix} Detections (threshold={threshold:.2f}):")
    if len(order) == 0:
        print("    (none)")
        return
    for idx in order:
        cls_id = int(l[idx])
        cls_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
        conf = float(s[idx]) * 100
        print(f"    {cls_name:20s}  {conf:5.1f}%  (class {cls_id})")


def _run_keras_detection(keras_lwdetr, image_float, resolution, num_select):
    """Run forward pass + postprocess on a single image.
    Returns (scores, labels, boxes) numpy arrays for the first image.
    """
    preprocessed = _preprocess(image_float, resolution)
    raw = keras_lwdetr(preprocessed, training=False)
    H, W = image_float.shape[:2]
    pp = K_PostProcess(num_select=num_select)
    scores, labels, boxes = pp(
        raw,
        ops.convert_to_tensor(np.array([[H, W]], dtype="float32")),
    )
    return (
        ops.convert_to_numpy(scores)[0],
        ops.convert_to_numpy(labels)[0],
        ops.convert_to_numpy(boxes)[0],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def coco_images():
    """Session-scoped: download all test COCO images as float32 [0,1]."""
    images = {}
    for name, info in COCO_IMAGES.items():
        arr = _download_coco_image(info["id"], info["url"])
        images[name] = arr.astype("float32") / 255.0
    return images


# ---------------------------------------------------------------------------
# Phase 1: Build Keras model from RFDETR, port PT weights, compare outputs
# ---------------------------------------------------------------------------


def _build_and_port_variant(variant_name):
    """Build PyTorch model, build Keras model via the RFDETR class,
    transfer weights.  Returns (pt_model, keras_facade).
    """
    config = MODEL_CONFIGS[variant_name]
    info = DETECTION_VARIANTS[variant_name]

    # 1. PyTorch model (auto-downloads weights)
    pt_model = config["pt_class"]()
    pt_model.model.model.eval()

    # 2. Keras RFDETR facade (skip pretrained download)
    facade = info["keras_cls"](pretrain_weights=None)

    # 3. Build all layers with training=True (needed for group_detr heads)
    res = facade.resolution
    dummy = np.ones((1, res, res, 3), dtype=np.float32) * 0.5
    facade.model.model(dummy, training=True)

    # 4. Transfer weights from PyTorch → Keras
    transfer_full_model_weights(pt_model, facade.model.model, config)

    return pt_model, facade


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPortingParity:
    """For each detection variant: port PT weights to Keras, verify output
    parity within 1e-4 on three COCO images, then save weights."""

    @pytest.fixture(
        scope="class",
        params=[
            v
            for v in DETECTION_VARIANTS
            if MODEL_CONFIGS.get(v, {}).get("pt_class") is not None
        ],
    )
    def variant(self, request, coco_images):
        """Class-scoped parameterised fixture: builds one variant at a time."""
        name = request.param
        print(f"\n{'=' * 60}")
        print(f"  Building variant: {name}")
        print(f"{'=' * 60}")

        pt_model, facade = _build_and_port_variant(name)

        yield {
            "name": name,
            "pt_model": pt_model,
            "facade": facade,
            "config": MODEL_CONFIGS[name],
            "images": coco_images,
        }

        # Teardown: free PyTorch model
        del pt_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Test 1: forward-pass parity on every COCO image ----------------

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_forward_parity(self, variant, image_name):
        """Raw logits and boxes must match within 1e-4 mean diff."""
        name = variant["name"]
        pt_model = variant["pt_model"]
        facade = variant["facade"]
        img = variant["images"][image_name]
        res = facade.resolution

        # Identical preprocessed input
        preprocessed = _preprocess(img, res)

        # PyTorch forward
        pt_input = torch.from_numpy(preprocessed).permute(0, 3, 1, 2)
        mask = torch.zeros((1, res, res), dtype=torch.bool)
        samples = NestedTensor(pt_input, mask)
        with torch.no_grad():
            pt_out = pt_model.model.model(samples)

        # Keras forward
        k_out = facade.model.model(preprocessed, training=False)

        # Compare logits
        pt_logits = pt_out["pred_logits"].cpu().numpy()
        k_logits = ops.convert_to_numpy(k_out["pred_logits"])
        diff_logits = np.abs(pt_logits - k_logits)

        # Compare boxes
        pt_boxes = pt_out["pred_boxes"].cpu().numpy()
        k_boxes = ops.convert_to_numpy(k_out["pred_boxes"])
        diff_boxes = np.abs(pt_boxes - k_boxes)

        print(
            f"\n  [{name}/{image_name}] Logits — "
            f"max: {diff_logits.max():.6e}, mean: {diff_logits.mean():.6e}"
        )
        print(
            f"  [{name}/{image_name}] Boxes  — "
            f"max: {diff_boxes.max():.6e}, mean: {diff_boxes.mean():.6e}"
        )

        assert diff_logits.mean() < 1e-4, (
            f"[{name}/{image_name}] Logits mean diff "
            f"{diff_logits.mean():.6e} > 1e-4"
        )
        assert diff_boxes.mean() < 1e-4, (
            f"[{name}/{image_name}] Boxes mean diff " f"{diff_boxes.mean():.6e} > 1e-4"
        )

    # ---- Test 2: detects expected objects on every COCO image -----------

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_detects_expected_objects(self, variant, image_name):
        """Keras model detects every expected COCO class above 0.3."""
        name = variant["name"]
        facade = variant["facade"]
        img = variant["images"][image_name]
        res = facade.resolution
        expected = COCO_IMAGES[image_name]["expected_classes"]

        scores, labels, _ = _run_keras_detection(
            facade.model.model, img, res, facade.model_config.num_select
        )

        _print_detections(scores, labels, f"{name}/{image_name}", threshold=0.3)

        detected = set(labels[scores > 0.3].tolist())
        for cls_id in expected:
            cls_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
            assert cls_id in detected, (
                f"[{name}/{image_name}] Expected '{cls_name}' "
                f"(class {cls_id}) not detected. Got: {detected}"
            )


# ---------------------------------------------------------------------------
# Phase 2: Save weights (only when all parity tests pass)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def save_weights_after_parity(request, coco_images):
    """After all Phase-1 tests pass, save .keras and .weights.h5 for every
    detection variant, then proceed to Phase 3 (h5 reload tests)."""
    yield  # wait for all tests to run first

    session = request.session
    if session.testsfailed > 0:
        print(
            f"\n[weight-save] {session.testsfailed} test(s) FAILED — "
            f"weights NOT saved."
        )
        return

    if not HAS_TORCH:
        print("\n[weight-save] PyTorch not available — skipping.")
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"\n{'=' * 60}")
    print("ALL PARITY TESTS PASSED — saving verified weights")
    print(f"{'=' * 60}")

    for name, info in DETECTION_VARIANTS.items():
        pt_cls = MODEL_CONFIGS.get(name, {}).get("pt_class")
        if pt_cls is None:
            print(f"  Skipping {name}: PT class unavailable")
            continue

        save_key = info["save_key"]
        keras_path = os.path.join(WEIGHTS_DIR, f"{save_key}.keras")
        h5_path = os.path.join(WEIGHTS_DIR, f"{save_key}.weights.h5")

        # Re-build and port if not yet done (each parameterised fixture
        # was class-scoped and already gone). Rebuilding is cheap compared
        # to the parity tests since we already verified correctness.
        print(f"\n  Building {name} for saving ...")
        try:
            _, facade = _build_and_port_variant(name)
            model = facade.model.model

            print(f"    Saving .keras  → {keras_path}")
            model.save(keras_path)

            print(f"    Saving .h5     → {h5_path}")
            model.save_weights(h5_path)

            # Free memory
            del facade
            gc.collect()
        except Exception as exc:
            print(f"    FAILED for {name}: {exc}")

    print(f"\n  Weights directory: {WEIGHTS_DIR}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Phase 3: Reload .h5 weights (no PyTorch) and re-run detection tests
# ---------------------------------------------------------------------------


class TestReloadH5Weights:
    """Load saved .weights.h5 into a fresh Keras RFDETR model (no PyTorch)
    and verify detection still works on all three COCO images."""

    @pytest.fixture(
        scope="class",
        params=list(DETECTION_VARIANTS.keys()),
    )
    def reloaded_model(self, request, coco_images):
        """Build a fresh Keras RFDETR, load .h5 weights, yield for tests."""
        name = request.param
        info = DETECTION_VARIANTS[name]
        save_key = info["save_key"]
        h5_path = os.path.join(WEIGHTS_DIR, f"{save_key}.weights.h5")

        if not os.path.exists(h5_path):
            pytest.skip(
                f"{h5_path} not found — Phase 2 may have been skipped or failed"
            )

        print(f"\n{'=' * 60}")
        print(f"  Reloading variant: {name} from .h5")
        print(f"{'=' * 60}")

        # Fresh Keras model (no PyTorch involved)
        facade = info["keras_cls"](pretrain_weights=None)

        # Build layers
        res = facade.resolution
        dummy = np.ones((1, res, res, 3), dtype=np.float32) * 0.5
        facade.model.model(dummy, training=True)

        # Load the verified .h5 weights
        facade.model.model.load_weights(h5_path)
        print(f"  Loaded weights from {h5_path}")

        yield {
            "name": name,
            "facade": facade,
            "images": coco_images,
        }

        del facade
        gc.collect()

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_h5_detects_expected_objects(self, reloaded_model, image_name):
        """After reloading .h5, the model detects expected objects."""
        name = reloaded_model["name"]
        facade = reloaded_model["facade"]
        img = reloaded_model["images"][image_name]
        res = facade.resolution
        expected = COCO_IMAGES[image_name]["expected_classes"]

        scores, labels, _ = _run_keras_detection(
            facade.model.model, img, res, facade.model_config.num_select
        )

        _print_detections(
            scores, labels, f"h5-reload/{name}/{image_name}", threshold=0.3
        )

        detected = set(labels[scores > 0.3].tolist())
        n_detections = int((scores > 0.3).sum())
        print(f"  [{name}/{image_name}] Total detections > 0.3: {n_detections}")

        for cls_id in expected:
            cls_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
            assert cls_id in detected, (
                f"[h5-reload/{name}/{image_name}] Expected '{cls_name}' "
                f"(class {cls_id}) not detected after .h5 reload. "
                f"Got: {detected}"
            )

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_h5_has_confident_detections(self, reloaded_model, image_name):
        """After reloading .h5, the model produces at least one
        detection above 0.3 confidence."""
        name = reloaded_model["name"]
        facade = reloaded_model["facade"]
        img = reloaded_model["images"][image_name]
        res = facade.resolution

        scores, labels, _ = _run_keras_detection(
            facade.model.model, img, res, facade.model_config.num_select
        )

        n = int((scores > 0.3).sum())
        assert n > 0, (
            f"[h5-reload/{name}/{image_name}] No detections > 0.3 " f"after .h5 reload"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
