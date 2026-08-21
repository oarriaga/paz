import gc
import io
import importlib
import os
import sys

import numpy as np
import pytest
from urllib.request import urlopen

# ---- path setup ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- Reference implementation guard -------------------------------------
try:
    import torch
    from PIL import Image

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---- Reference RF-DETR imports (detection only) -------------------------
if HAS_TORCH:
    # Import rfdetr for its side effect only; the fallback adds the vendored
    # copy under examples/ to sys.path so NestedTensor (below) resolves.
    try:
        importlib.import_module("rfdetr")
    except ImportError:
        rfdetr_path = os.path.abspath(
            os.path.join(
                current_dir,
                "../../../../examples/"
                "rf-detr_original_pytorch_implementation",
            )
        )
        if rfdetr_path not in sys.path:
            sys.path.insert(0, rfdetr_path)
        importlib.import_module("rfdetr")

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

# ---- Keras RF-DETR imports (detection only) ------------------------------
from paz.models.detection.dino_v2_object_detection.detr import (
    RFDETRBase as K_RFDETRBase,
    RFDETRNano as K_RFDETRNano,
    RFDETRSmall as K_RFDETRSmall,
    RFDETRMedium as K_RFDETRMedium,
    RFDETRLarge as K_RFDETRLarge,
    RFDETRXLarge as K_RFDETRXLarge,
    RFDETR2XLarge as K_RFDETR2XLarge,
)
import functools

from paz.models.detection.dino_v2_object_detection.main import (
    post_process,
)
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    apply_lwdetr,
)
from paz.models.detection.dino_v2_object_detection.utils.coco_classes import (  # fmt: skip
    COCO_CLASSES,
)

# Weight-transfer utilities
if HAS_TORCH:
    from paz.models.detection.dino_v2_object_detection.models.lwdetr.test_lwdetr_with_real_weights import (  # fmt: skip
        transfer_full_model_weights,
        MODEL_CONFIGS,
    )

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
    "RFDETR2XLarge": {
        "keras_cls": K_RFDETR2XLarge,
        "save_key": "rfdetr_2xlarge",
    },
}

IMAGENET_MEANS = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STDS = np.array([0.229, 0.224, 0.225], dtype="float32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def download_coco_image(image_id, url):
    ensure_cache_dir()
    cached = os.path.join(CACHE_DIR, f"coco_val_{image_id}.npy")
    if os.path.exists(cached):
        pixels = np.load(cached)
    else:
        print(f"  Downloading COCO image {image_id} ...")
        data = urlopen(url).read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        pixels = np.array(image, dtype=np.uint8)
        np.save(cached, pixels)
    return pixels


def preprocess_image(image_float, resolution):
    normed = (image_float - IMAGENET_MEANS) / IMAGENET_STDS
    t = ops.convert_to_tensor(normed[np.newaxis], dtype="float32")
    resized = ops.image.resize(t, (resolution, resolution))
    return ops.convert_to_numpy(resized)


def print_detections(scores, labels, header="", threshold=0.3):
    keep = scores > threshold
    kept_scores, kept_labels = scores[keep], labels[keep]
    order = np.argsort(-kept_scores)
    prefix = f"  [{header}]" if header else "  "
    print(f"{prefix} Detections (threshold={threshold:.2f}):")
    if len(order) == 0:
        print("    (none)")
    for index in order:
        class_id = int(kept_labels[index])
        class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
        confidence = float(kept_scores[index]) * 100
        print(f"    {class_name:20s}  {confidence:5.1f}%  (class {class_id})")


def run_keras_detection(keras_lwdetr, image_float, resolution, num_select):
    preprocessed = preprocess_image(image_float, resolution)
    raw = apply_lwdetr(keras_lwdetr, preprocessed, training=False)
    height, width = image_float.shape[:2]
    sizes = np.array([[height, width]], dtype="float32")
    postprocess = functools.partial(post_process, num_select=num_select)
    outputs = postprocess(raw, ops.convert_to_tensor(sizes))
    return [ops.convert_to_numpy(output)[0] for output in outputs]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def coco_images():
    images = {}
    for name, info in COCO_IMAGES.items():
        pixels = download_coco_image(info["id"], info["url"])
        images[name] = pixels.astype("float32") / 255.0
    return images


# ---------------------------------------------------------------------------
# Phase 1: Build Keras model, port reference weights, compare outputs
# ---------------------------------------------------------------------------


def build_and_port_variant(variant_name):
    config = MODEL_CONFIGS[variant_name]
    info = DETECTION_VARIANTS[variant_name]

    # 1. Reference model (auto-downloads weights)
    pt_model = config["pt_class"]()
    pt_model.model.model.eval()
    pt_model.model.model.cpu()

    # 2. Keras RF-DETR facade (skip pretrained download)
    facade = info["keras_cls"](pretrain_weights=None)

    # 3. Build all layers with training=True (needed for group_detr heads)
    resolution = facade.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype=np.float32) * 0.5
    apply_lwdetr(facade.model.model, dummy, training=True)

    # 4. Transfer weights from reference model to Keras
    transfer_full_model_weights(pt_model, facade.model.model, config)

    return pt_model, facade


@pytest.fixture(
    scope="class",
    params=[
        v
        for v in DETECTION_VARIANTS
        if MODEL_CONFIGS.get(v, {}).get("pt_class") is not None
    ],
)
def variant(request, coco_images):
    name = request.param
    print(f"\n{'=' * 60}")
    print(f"  Building variant: {name}")
    print(f"{'=' * 60}")

    pt_model, facade = build_and_port_variant(name)

    yield {
        "name": name,
        "pt_model": pt_model,
        "facade": facade,
        "config": MODEL_CONFIGS[name],
        "images": coco_images,
    }

    # Teardown: free reference model
    del pt_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Test 1: forward-pass parity on every COCO image --------------------


@pytest.mark.skipif(not HAS_TORCH, reason="Reference library not installed")
@pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
def run_reference_forward(pt_model, preprocessed, resolution):
    pt_input = torch.from_numpy(preprocessed).permute(0, 3, 1, 2)
    mask = torch.zeros((1, resolution, resolution), dtype=torch.bool)
    with torch.no_grad():
        outputs = pt_model.model.model(NestedTensor(pt_input, mask))
    return outputs


def compare_parity_field(pt_out, k_out, key, tag, label):
    reference = pt_out[key].cpu().numpy()
    difference = np.abs(reference - ops.convert_to_numpy(k_out[key]))
    summary = f"max: {difference.max():.6e}, mean: {difference.mean():.6e}"
    print(f"\n  [{tag}] {label} - {summary}")
    message = f"[{tag}] {label} mean diff {difference.mean():.6e} > 1e-4"
    assert difference.mean() < 1e-4, message


def test_forward_parity(variant, image_name):
    facade = variant["facade"]
    resolution = facade.resolution
    image = variant["images"][image_name]
    # Both models see the identical preprocessed input.
    preprocessed = preprocess_image(image, resolution)
    args = (variant["pt_model"], preprocessed, resolution)
    pt_out = run_reference_forward(*args)
    k_out = apply_lwdetr(facade.model.model, preprocessed, training=False)
    tag = f"{variant['name']}/{image_name}"
    compare_parity_field(pt_out, k_out, "pred_logits", tag, "Logits")
    compare_parity_field(pt_out, k_out, "pred_boxes", tag, "Boxes")


# ---- Test 2: detects expected objects on every COCO image ---------------


@pytest.mark.skipif(not HAS_TORCH, reason="Reference library not installed")
@pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
def test_detects_expected_objects(variant, image_name):
    name = variant["name"]
    facade = variant["facade"]
    image = variant["images"][image_name]
    resolution = facade.resolution
    expected = COCO_IMAGES[image_name]["expected_classes"]

    scores, labels, _ = run_keras_detection(
        facade.model.model, image, resolution, facade.model_config.num_select
    )

    print_detections(scores, labels, f"{name}/{image_name}", threshold=0.3)

    detected = set(labels[scores > 0.3].tolist())
    for cls_id in expected:
        cls_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
        assert cls_id in detected, (
            f"[{name}/{image_name}] Expected '{cls_name}' "
            f"(class {cls_id}) not detected. Got: {detected}"
        )


# ---------------------------------------------------------------------------
# Phase 2: Save verified weights to disk
# ---------------------------------------------------------------------------


def save_verified_variant(name, info):
    save_key = info["save_key"]
    keras_path = os.path.join(WEIGHTS_DIR, f"{save_key}.keras")
    h5_path = os.path.join(WEIGHTS_DIR, f"{save_key}.weights.h5")
    # Each parameterised fixture was class-scoped and is already gone, so
    # rebuild; that is cheap now that parity has been verified.
    print(f"\n  Building {name} for saving ...")
    try:
        facade = build_and_port_variant(name)[1]
        print(f"    Saving .keras  -> {keras_path}")
        facade.model.model.save(keras_path)
        print(f"    Saving .h5     -> {h5_path}")
        facade.model.model.save_weights(h5_path)
        del facade
        gc.collect()
    except Exception as error:
        print(f"    FAILED for {name}: {error}")


def save_verified_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"\n{'=' * 60}")
    print("ALL PARITY TESTS PASSED - saving verified weights")
    print(f"{'=' * 60}")
    for name, info in DETECTION_VARIANTS.items():
        if MODEL_CONFIGS.get(name, {}).get("pt_class") is None:
            print(f"  Skipping {name}: reference class unavailable")
        else:
            save_verified_variant(name, info)
    print(f"\n  Weights directory: {WEIGHTS_DIR}")
    print(f"{'=' * 60}\n")


# coco_images is requested so the session fixture outlives the cached image
# downloads the parity tests depend on.
@pytest.fixture(scope="session", autouse=True)
def save_weights_after_parity(request, coco_images):
    yield  # wait for all tests to run first
    failed = request.session.testsfailed
    if failed > 0:
        print(f"\n[weight-save] {failed} test(s) FAILED - weights NOT saved.")
    elif not HAS_TORCH:
        print("\n[weight-save] Reference library not available - skipping.")
    else:
        save_verified_weights()


# ---------------------------------------------------------------------------
# Phase 3: Reload .h5 weights and re-run detection tests
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="class",
    params=list(DETECTION_VARIANTS.keys()),
)
def reloaded_model(request, coco_images):
    name = request.param
    info = DETECTION_VARIANTS[name]
    save_key = info["save_key"]
    h5_path = os.path.join(WEIGHTS_DIR, f"{save_key}.weights.h5")

    if not os.path.exists(h5_path):
        pytest.skip(f"{h5_path} not found - Phase 2 skipped or failed")

    print(f"\n{'=' * 60}")
    print(f"  Reloading variant: {name} from .h5")
    print(f"{'=' * 60}")

    # Fresh Keras model (no reference library required); one forward pass
    # materialises every layer before the verified .h5 weights load.
    facade = info["keras_cls"](pretrain_weights=None)
    resolution = facade.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype=np.float32) * 0.5
    apply_lwdetr(facade.model.model, dummy, training=True)
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
def test_h5_detects_expected_objects(reloaded_model, image_name):
    name = reloaded_model["name"]
    facade = reloaded_model["facade"]
    image = reloaded_model["images"][image_name]
    resolution = facade.resolution
    expected = COCO_IMAGES[image_name]["expected_classes"]

    scores, labels, _ = run_keras_detection(
        facade.model.model, image, resolution, facade.model_config.num_select
    )

    print_detections(
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
def test_h5_has_confident_detections(reloaded_model, image_name):
    name = reloaded_model["name"]
    facade = reloaded_model["facade"]
    image = reloaded_model["images"][image_name]
    resolution = facade.resolution

    scores, labels, _ = run_keras_detection(
        facade.model.model, image, resolution, facade.model_config.num_select
    )

    n = int((scores > 0.3).sum())
    assert n > 0, (
        f"[h5-reload/{name}/{image_name}] No detections > 0.3 "
        f"after .h5 reload"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
