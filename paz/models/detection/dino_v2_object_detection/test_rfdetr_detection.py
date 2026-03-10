import gc
import io
import os

import numpy as np
import pytest
from PIL import Image
from urllib.request import urlopen

# ── Keras imports (always available) ──────────────────────────────────────────
from paz.models.detection.dino_v2_object_detection.detr import (
    RFDETR as K_RFDETR,
    RFDETRBase as K_RFDETRBase,
    RFDETRNano as K_RFDETRNano,
    RFDETRSmall as K_RFDETRSmall,
    RFDETRMedium as K_RFDETRMedium,
    RFDETRLarge as K_RFDETRLarge,
    RFDETRXLarge as K_RFDETRXLarge,
    RFDETR2XLarge as K_RFDETR2XLarge,
    VARIANT_REGISTRY as K_REGISTRY,
)
from paz.models.detection.dino_v2_object_detection.utils.coco_classes import COCO_CLASSES

# ── Reference implementation (optional — for parity comparison) ──────────────────
try:
    from rfdetr import (
        RFDETRBase as PT_RFDETRBase,
        RFDETRNano as PT_RFDETRNano,
        RFDETRSmall as PT_RFDETRSmall,
        RFDETRMedium as PT_RFDETRMedium,
        RFDETRLarge as PT_RFDETRLarge,
    )

    HAS_PT = True
except ImportError:
    HAS_PT = False

needs_pt = pytest.mark.skipif(not HAS_PT, reason="Reference library not installed")

# ── Constants ─────────────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".test_cache")

COCO_IMAGES = {
    "cats": {
        "id": "000000039769",
        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "expected_classes": {17},
    },
    "bear": {
        "id": "000000000285",
        "url": "http://images.cocodataset.org/val2017/000000000285.jpg",
        "expected_classes": {23},
    },
    "kitchen": {
        "id": "000000037777",
        "url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "expected_classes": {82},
    },
}

# Detection variants.  ``pt`` is None when no reference equivalent exists.
VARIANTS = {
    "nano": {
        "keras": K_RFDETRNano,
        "pt": PT_RFDETRNano if HAS_PT else None,
        "res": 384,
        "coco": True,
    },
    "small": {
        "keras": K_RFDETRSmall,
        "pt": PT_RFDETRSmall if HAS_PT else None,
        "res": 512,
        "coco": True,
    },
    "medium": {
        "keras": K_RFDETRMedium,
        "pt": PT_RFDETRMedium if HAS_PT else None,
        "res": 576,
        "coco": True,
    },
    "base": {
        "keras": K_RFDETRBase,
        "pt": PT_RFDETRBase if HAS_PT else None,
        "res": 560,
        "coco": True,
    },
    "large": {
        "keras": K_RFDETRLarge,
        "pt": PT_RFDETRLarge if HAS_PT else None,
        "res": 704,
        "coco": True,
    },
    "xlarge": {
        "keras": K_RFDETRXLarge,
        "pt": None,
        "res": 700,
        "coco": False,
    },
    "2xlarge": {
        "keras": K_RFDETR2XLarge,
        "pt": None,
        "res": 880,
        "coco": False,
    },
}

# Handy name lists
ALL_NAMES = list(VARIANTS.keys())
COCO_NAMES = [k for k, v in VARIANTS.items() if v["coco"]]
PARITY_NAMES = [k for k, v in VARIANTS.items() if v["pt"] is not None]

# Tolerance settings for cross-implementation comparison
SCORE_ATOL = 0.05
BOX_ATOL = 15.0
TOP_K = 5


# ── Helpers ───────────────────────────────────────────────────────────────────


def _download_image(name):
    """Download and cache a COCO test image as uint8 ``(H, W, 3)``.

    Args:
        name (str): Key in ``COCO_IMAGES``.

    Returns:
        np.ndarray: ``(H, W, 3)`` uint8 RGB array.
    """
    info = COCO_IMAGES[name]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"coco_{info['id']}.npy")
    if os.path.exists(path):
        return np.load(path)
    data = urlopen(info["url"]).read()
    arr = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    np.save(path, arr)
    return arr


def _pt_to_dict(det):
    """Convert a reference ``Detections`` object to a standard dict."""
    return {"boxes": det.xyxy, "scores": det.confidence, "labels": det.class_id}


def _sort_desc(result):
    """Sort a detection result dict by descending score."""
    idx = np.argsort(-result["scores"])
    return {k: result[k][idx] for k in ("boxes", "scores", "labels")}


def _assert_top_k_close(
    keras_res, pt_res, k=TOP_K, score_atol=SCORE_ATOL, box_atol=BOX_ATOL
):
    """Assert the top-K detections agree between two result dicts.

    After selecting top-K by score, both sets are re-sorted by
    ``(label, box_x1, box_y1)`` so that minor score-ordering differences
    do not cause spurious label-mismatch failures.

    Args:
        keras_res (dict): Keras detection results.
        pt_res (dict): Reference detection results.
        k (int): Number of top detections to compare.
        score_atol (float): Absolute tolerance for scores.
        box_atol (float): Absolute tolerance for box coordinates.
    """
    ks = _sort_desc(keras_res)
    ps = _sort_desc(pt_res)
    n = min(k, len(ks["scores"]), len(ps["scores"]))
    assert n > 0, "No detections from either model"
    # Take top-K, then re-sort deterministically by (label, x1, y1)
    k_idx = np.lexsort((ks["boxes"][:n, 1], ks["boxes"][:n, 0], ks["labels"][:n]))
    p_idx = np.lexsort((ps["boxes"][:n, 1], ps["boxes"][:n, 0], ps["labels"][:n]))
    for key in ("scores", "labels", "boxes"):
        ks[key] = ks[key][:n][k_idx]
        ps[key] = ps[key][:n][p_idx]
    np.testing.assert_allclose(
        ks["scores"], ps["scores"], atol=score_atol, err_msg="scores"
    )
    np.testing.assert_array_equal(ks["labels"], ps["labels"], err_msg="labels")
    np.testing.assert_allclose(ks["boxes"], ps["boxes"], atol=box_atol, err_msg="boxes")


def _build_and_compare(keras_cls, pt_cls, images):
    """Build both models, predict on every image, and compare top-K."""
    k_model = keras_cls()
    pt_model = pt_cls()
    try:
        for name, img in images.items():
            k_res = k_model.predict(img)[0]
            pt_res = _pt_to_dict(pt_model.predict(img))
            _assert_top_k_close(k_res, pt_res)
    finally:
        del k_model, pt_model
        gc.collect()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def coco_images():
    """All COCO test images, cached to disk."""
    return {n: _download_image(n) for n in COCO_IMAGES}


@pytest.fixture(scope="module")
def cats_image(coco_images):
    """Single 'cats' test image."""
    return coco_images["cats"]


@pytest.fixture(scope="module")
def keras_nano():
    """Keras ``RFDETRNano`` loaded with pretrained weights."""
    m = K_RFDETRNano()
    yield m
    del m
    gc.collect()


@pytest.fixture(scope="module")
def pt_nano():
    """Reference ``RFDETRNano`` (skipped when the library is not installed)."""
    if not HAS_PT:
        pytest.skip("rfdetr not installed")
    m = PT_RFDETRNano()
    yield m
    del m
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — Keras properties (no weights needed, fast)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", ALL_NAMES)
def test_resolution(name):
    """Resolution matches the documented config value for each variant."""
    m = VARIANTS[name]["keras"](pretrain_weights=None)
    assert m.resolution == VARIANTS[name]["res"]


@pytest.mark.parametrize("name", COCO_NAMES)
def test_class_names_coco(name):
    """COCO variants return standard COCO class names."""
    m = VARIANTS[name]["keras"](pretrain_weights=None)
    assert m.class_names == COCO_CLASSES


def test_variant_registry_complete():
    """All seven detection variant names are in the Keras registry."""
    expected = {
        "RFDETRNano",
        "RFDETRSmall",
        "RFDETRMedium",
        "RFDETRBase",
        "RFDETRLarge",
        "RFDETRXLarge",
        "RFDETR2XLarge",
    }
    assert expected <= set(K_REGISTRY.keys())


def test_all_detection_variants_are_rfdetr_subclass():
    """Every detection variant inherits from RFDETR."""
    for name in ALL_NAMES:
        assert issubclass(VARIANTS[name]["keras"], K_RFDETR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — Keras predict format (Nano, loaded weights)
# ═══════════════════════════════════════════════════════════════════════════════


def test_predict_returns_list_of_dicts(keras_nano, cats_image):
    """predict() returns a list of dicts with required keys."""
    results = keras_nano.predict(cats_image)
    assert isinstance(results, list) and len(results) == 1
    assert {"boxes", "scores", "labels"} <= set(results[0].keys())


def test_predict_scores_in_range(keras_nano, cats_image):
    """All scores lie in [0, 1]."""
    s = keras_nano.predict(cats_image)[0]["scores"]
    assert np.all((s >= 0) & (s <= 1))


def test_predict_boxes_positive(keras_nano, cats_image):
    """Box coordinates (xyxy) are non-negative."""
    b = keras_nano.predict(cats_image)[0]["boxes"]
    assert b.size == 0 or np.all(b >= 0)


def test_predict_labels_integer(keras_nano, cats_image):
    """Labels are integers."""
    lbl = keras_nano.predict(cats_image)[0]["labels"]
    assert np.issubdtype(lbl.dtype, np.integer)


def test_predict_batch(keras_nano, cats_image):
    """Batch of two images returns two result dicts."""
    batch = np.stack([cats_image, cats_image])
    assert len(keras_nano.predict(batch)) == 2


def test_threshold_filtering(keras_nano, cats_image):
    """Higher threshold → fewer (or equal) detections."""
    lo = keras_nano.predict(cats_image, threshold=0.1)[0]
    hi = keras_nano.predict(cats_image, threshold=0.8)[0]
    assert len(lo["scores"]) >= len(hi["scores"])


def test_uint8_float_equivalence(keras_nano, cats_image):
    """uint8 and float32 [0,1] inputs give identical results."""
    r1 = keras_nano.predict(cats_image)[0]
    r2 = keras_nano.predict(cats_image.astype("float32") / 255.0)[0]
    np.testing.assert_allclose(r1["scores"], r2["scores"], atol=1e-5)
    np.testing.assert_allclose(r1["boxes"], r2["boxes"], atol=0.5)


@pytest.mark.parametrize("img_name", list(COCO_IMAGES.keys()))
def test_expected_class_detected(keras_nano, coco_images, img_name):
    """Keras Nano detects at least one expected class per image."""
    r = keras_nano.predict(coco_images[img_name], threshold=0.3)[0]
    expected = COCO_IMAGES[img_name]["expected_classes"]
    detected = set(r["labels"].tolist())
    assert expected & detected, f"Expected {expected}, detected {detected}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — Nano parity: Keras vs reference (module fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


@needs_pt
def test_resolution_parity_nano(keras_nano, pt_nano):
    """Keras and reference Nano report the same resolution."""
    assert keras_nano.resolution == pt_nano.model_config.resolution


@needs_pt
def test_class_names_parity_nano(keras_nano, pt_nano):
    """Keras and reference Nano report the same class names."""
    assert keras_nano.class_names == pt_nano.class_names


@needs_pt
@pytest.mark.parametrize("img_name", list(COCO_IMAGES.keys()))
def test_predict_parity_nano(keras_nano, pt_nano, coco_images, img_name):
    """Nano top-K detections match between Keras and the reference."""
    k_res = keras_nano.predict(coco_images[img_name])[0]
    pt_res = _pt_to_dict(pt_nano.predict(coco_images[img_name]))
    _assert_top_k_close(k_res, pt_res)


@needs_pt
@pytest.mark.parametrize("img_name", list(COCO_IMAGES.keys()))
def test_same_expected_class_nano(keras_nano, pt_nano, coco_images, img_name):
    """Both Keras and reference Nano detect the expected class."""
    expected = COCO_IMAGES[img_name]["expected_classes"]
    k_labels = set(
        keras_nano.predict(coco_images[img_name], threshold=0.3)[0]["labels"].tolist()
    )
    p_labels = set(
        pt_nano.predict(coco_images[img_name], threshold=0.3).class_id.tolist()
    )
    assert expected & k_labels, f"Keras missing {expected}, got {k_labels}"
    assert expected & p_labels, f"PT missing {expected}, got {p_labels}"


@needs_pt
@pytest.mark.parametrize("img_name", list(COCO_IMAGES.keys()))
def test_detection_count_similar_nano(keras_nano, pt_nano, coco_images, img_name):
    """Keras and reference Nano produce a similar detection count."""
    k_n = len(keras_nano.predict(coco_images[img_name], threshold=0.5)[0]["scores"])
    p_n = len(pt_nano.predict(coco_images[img_name], threshold=0.5).confidence)
    assert abs(k_n - p_n) <= 3, f"Keras={k_n}, PT={p_n}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — Multi-variant parity (builds models per invocation)
# ═══════════════════════════════════════════════════════════════════════════════


@needs_pt
@pytest.mark.parametrize("variant", PARITY_NAMES)
def test_variant_parity_all_images(variant, coco_images):
    """Top-K predictions match across all test images for each variant."""
    v = VARIANTS[variant]
    _build_and_compare(v["keras"], v["pt"], coco_images)


@needs_pt
@pytest.mark.parametrize("variant", PARITY_NAMES)
def test_variant_expected_classes(variant, coco_images):
    """Both Keras and reference detect expected classes for each variant."""
    v = VARIANTS[variant]
    k_model = v["keras"]()
    pt_model = v["pt"]()
    try:
        for img_name, img in coco_images.items():
            expected = COCO_IMAGES[img_name]["expected_classes"]
            k_set = set(k_model.predict(img, threshold=0.3)[0]["labels"].tolist())
            p_set = set(pt_model.predict(img, threshold=0.3).class_id.tolist())
            assert (
                expected & k_set
            ), f"Keras {variant}/{img_name}: {expected} vs {k_set}"
            assert expected & p_set, f"PT {variant}/{img_name}: {expected} vs {p_set}"
    finally:
        del k_model, pt_model
        gc.collect()


@needs_pt
@pytest.mark.parametrize("variant", PARITY_NAMES)
def test_variant_resolution_parity(variant):
    """Resolution matches between Keras and reference for each variant."""
    v = VARIANTS[variant]
    k = v["keras"](pretrain_weights=None)
    pt = v["pt"]()
    try:
        assert k.resolution == pt.model_config.resolution
    finally:
        del k, pt
        gc.collect()


@needs_pt
@pytest.mark.parametrize("variant", PARITY_NAMES)
def test_variant_class_names_parity(variant):
    """Class names match between Keras and reference for each COCO variant."""
    v = VARIANTS[variant]
    k = v["keras"](pretrain_weights=None)
    pt = v["pt"]()
    try:
        assert k.class_names == pt.class_names
    finally:
        del k, pt
        gc.collect()
