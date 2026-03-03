import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Keras implementation (the code under test)
# ---------------------------------------------------------------------------
from paz.models.detection.dino_v2_object_detection.config import (
    ModelConfig as KerasModelConfig,
    TrainConfig as KerasTrainConfig,
    SegmentationTrainConfig as KerasSegTrainConfig,
    RFDETRBaseConfig as KerasBaseConfig,
    RFDETRNanoConfig as KerasNanoConfig,
    RFDETRSmallConfig as KerasSmallConfig,
    RFDETRMediumConfig as KerasMediumConfig,
    RFDETRLargeConfig as KerasLargeConfig,
    RFDETRXLargeConfig as KerasXLargeConfig,
    RFDETR2XLargeConfig as Keras2XLargeConfig,
    RFDETRSegPreviewConfig as KerasSegPreviewConfig,
    RFDETRSegNanoConfig as KerasSegNanoConfig,
    RFDETRSegSmallConfig as KerasSegSmallConfig,
    RFDETRSegMediumConfig as KerasSegMediumConfig,
    RFDETRSegLargeConfig as KerasSegLargeConfig,
    RFDETRSegXLargeConfig as KerasSegXLargeConfig,
    RFDETRSeg2XLargeConfig as KerasSeg2XLargeConfig,
)
from paz.models.detection.dino_v2_object_detection.detr import (
    RFDETR as KerasRFDETR,
    RFDETRBase as KerasRFDETRBase,
    RFDETRNano as KerasRFDETRNano,
    RFDETRSmall as KerasRFDETRSmall,
    RFDETRMedium as KerasRFDETRMedium,
    RFDETRLarge as KerasRFDETRLarge,
    RFDETRXLarge as KerasRFDETRXLarge,
    RFDETR2XLarge as KerasRFDETR2XLarge,
    RFDETRSegPreview as KerasRFDETRSegPreview,
    RFDETRSegNano as KerasRFDETRSegNano,
    RFDETRSegSmall as KerasRFDETRSegSmall,
    RFDETRSegMedium as KerasRFDETRSegMedium,
    RFDETRSegLarge as KerasRFDETRSegLarge,
    RFDETRSegXLarge as KerasRFDETRSegXLarge,
    RFDETRSeg2XLarge as KerasRFDETRSeg2XLarge,
    VARIANT_REGISTRY as KerasVariantRegistry,
    _COCODataLoader,
)
from paz.models.detection.dino_v2_object_detection.utils.coco_classes import (
    COCO_CLASSES as KerasCOCO,
)
from paz.models.detection.dino_v2_object_detection.utils.utils import (
    ModelEma as KerasModelEma,
    BestMetricHolder as KerasBestMetricHolder,
)
from paz.models.detection.dino_v2_object_detection.utils.early_stopping import (
    EarlyStoppingCallback as KerasEarlyStoppingCallback,
)
from paz.models.detection.dino_v2_object_detection.utils.metrics import (
    MetricsPlotSink as KerasPlotSink,
    MetricsTensorBoardSink as KerasTBSink,
    MetricsWandBSink as KerasWBSink,
)
from paz.models.detection.dino_v2_object_detection.engine import (
    build_lr_lambda as keras_build_lr_lambda,
)

# ---------------------------------------------------------------------------
# PyTorch implementation (reference)
# ---------------------------------------------------------------------------
import sys

_PT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..",
        "examples", "rf-detr_original_pytorch_implementation",
    )
)
if _PT_ROOT not in sys.path:
    sys.path.insert(0, _PT_ROOT)

from rfdetr.config import (
    ModelConfig as PTModelConfig,
    TrainConfig as PTTrainConfig,
    SegmentationTrainConfig as PTSegTrainConfig,
    RFDETRBaseConfig as PTBaseConfig,
    RFDETRNanoConfig as PTNanoConfig,
    RFDETRSmallConfig as PTSmallConfig,
    RFDETRMediumConfig as PTMediumConfig,
    RFDETRLargeConfig as PTLargeConfig,
    RFDETRSegPreviewConfig as PTSegPreviewConfig,
    RFDETRSegNanoConfig as PTSegNanoConfig,
    RFDETRSegSmallConfig as PTSegSmallConfig,
    RFDETRSegMediumConfig as PTSegMediumConfig,
    RFDETRSegLargeConfig as PTSegLargeConfig,
    RFDETRSegXLargeConfig as PTSegXLargeConfig,
    RFDETRSeg2XLargeConfig as PTSeg2XLargeConfig,
)
from rfdetr.detr import (
    RFDETR as PTRFDETR,
    RFDETRBase as PTRFDETRBase,
    RFDETRNano as PTRFDETRNano,
    RFDETRSmall as PTRFDETRSmall,
    RFDETRMedium as PTRFDETRMedium,
    RFDETRLarge as PTRFDETRLarge,
    RFDETRSegPreview as PTRFDETRSegPreview,
    RFDETRSegNano as PTRFDETRSegNano,
    RFDETRSegSmall as PTRFDETRSegSmall,
    RFDETRSegMedium as PTRFDETRSegMedium,
    RFDETRSegLarge as PTRFDETRSegLarge,
    RFDETRSegXLarge as PTRFDETRSegXLarge,
    RFDETRSeg2XLarge as PTRFDETRSeg2XLarge,
)
from rfdetr.util.coco_classes import COCO_CLASSES as PTCOCO


# ===================================================================
# Helpers
# ===================================================================

def _keras_config_to_dict(cfg):
    """Convert a Keras dataclass config to a flat dict."""
    return asdict(cfg)


def _pt_config_to_dict(cfg):
    """Convert a Pydantic (PyTorch) config to a flat dict."""
    return cfg.dict()


def _shared_config_keys(keras_dict, pt_dict):
    """Return set of keys present in both dicts."""
    return set(keras_dict.keys()) & set(pt_dict.keys())


def _make_dummy_coco_dataset(tmpdir, num_images=3, num_classes=2):
    """Create a minimal COCO-format dataset in *tmpdir*."""
    categories = [
        {"id": i + 1, "name": f"class_{i}", "supercategory": "object"}
        for i in range(num_classes)
    ]
    images = []
    annotations = []
    ann_id = 1
    for img_id in range(1, num_images + 1):
        fname = f"img_{img_id:04d}.jpg"
        images.append({"id": img_id, "file_name": fname, "width": 64, "height": 64})
        # Write a tiny JPEG
        _write_dummy_image(os.path.join(tmpdir, fname), 64, 64)
        # One annotation per image
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "iscrowd": 0,
        })
        ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(os.path.join(tmpdir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)
    return coco


def _write_dummy_image(path, w, h):
    """Write a tiny random JPEG to *path*."""
    from PIL import Image as PILImage

    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(path, "JPEG")


def _make_dataset_dir(num_classes=2):
    """Create a temp dataset dir with train/valid splits."""
    tmpdir = tempfile.mkdtemp(prefix="rfdetr_test_")
    for split in ("train", "valid"):
        split_dir = os.path.join(tmpdir, split)
        os.makedirs(split_dir, exist_ok=True)
        _make_dummy_coco_dataset(split_dir, num_images=4, num_classes=num_classes)
    return tmpdir


# Fixture to mock Model so no actual heavy model is built
@pytest.fixture
def mock_keras_model():
    """Patch the Keras Model class to avoid building a real network."""
    mock_model_instance = MagicMock()
    mock_model_instance.class_names = None
    mock_model_instance.resolution = 560
    mock_model_instance.config = KerasBaseConfig()
    mock_model_instance.model = MagicMock()
    mock_model_instance.model.weights = []
    mock_model_instance.model.trainable_variables = []
    mock_model_instance.reinitialize_detection_head = MagicMock()

    with patch(
        "paz.models.detection.dino_v2_object_detection.detr.Model",
        return_value=mock_model_instance,
    ) as mock_cls:
        yield mock_cls, mock_model_instance


@pytest.fixture
def mock_pt_model():
    """Patch the PyTorch Model class to avoid building a real network."""
    mock = MagicMock()
    mock.class_names = None
    mock.resolution = 560
    mock.model = MagicMock()
    mock.inference_model = None

    with patch(
        "rfdetr.detr.Model",
        return_value=mock,
    ) as mock_cls, patch(
        "rfdetr.detr.download_pretrain_weights",
    ):
        yield mock_cls, mock


# ===================================================================
# 1. Config parity tests
# ===================================================================

# Detection variant configs
DETECTION_CONFIG_PAIRS = [
    (KerasBaseConfig, PTBaseConfig, "base"),
    (KerasNanoConfig, PTNanoConfig, "nano"),
    (KerasSmallConfig, PTSmallConfig, "small"),
    (KerasMediumConfig, PTMediumConfig, "medium"),
    (KerasLargeConfig, PTLargeConfig, "large"),
]

# Segmentation variant configs
SEG_CONFIG_PAIRS = [
    (KerasSegPreviewConfig, PTSegPreviewConfig, "seg_preview"),
    (KerasSegNanoConfig, PTSegNanoConfig, "seg_nano"),
    (KerasSegSmallConfig, PTSegSmallConfig, "seg_small"),
    (KerasSegMediumConfig, PTSegMediumConfig, "seg_medium"),
    (KerasSegLargeConfig, PTSegLargeConfig, "seg_large"),
    (KerasSegXLargeConfig, PTSegXLargeConfig, "seg_xlarge"),
    (KerasSeg2XLargeConfig, PTSeg2XLargeConfig, "seg_2xlarge"),
]


@pytest.mark.parametrize("keras_cls,pt_cls,name", DETECTION_CONFIG_PAIRS)
def test_detection_config_resolution_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.resolution == p.resolution


@pytest.mark.parametrize("keras_cls,pt_cls,name", DETECTION_CONFIG_PAIRS)
def test_detection_config_hidden_dim_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.hidden_dim == p.hidden_dim


@pytest.mark.parametrize("keras_cls,pt_cls,name", DETECTION_CONFIG_PAIRS)
def test_detection_config_dec_layers_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.dec_layers == p.dec_layers


@pytest.mark.parametrize("keras_cls,pt_cls,name", DETECTION_CONFIG_PAIRS)
def test_detection_config_patch_size_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.patch_size == p.patch_size


@pytest.mark.parametrize("keras_cls,pt_cls,name", DETECTION_CONFIG_PAIRS)
def test_detection_config_num_windows_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.num_windows == p.num_windows


@pytest.mark.parametrize("keras_cls,pt_cls,name", DETECTION_CONFIG_PAIRS)
def test_detection_config_encoder_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.encoder == p.encoder


@pytest.mark.parametrize("keras_cls,pt_cls,name", SEG_CONFIG_PAIRS)
def test_seg_config_resolution_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.resolution == p.resolution


@pytest.mark.parametrize("keras_cls,pt_cls,name", SEG_CONFIG_PAIRS)
def test_seg_config_segmentation_flag(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.segmentation_head is True
    assert p.segmentation_head is True


@pytest.mark.parametrize("keras_cls,pt_cls,name", SEG_CONFIG_PAIRS)
def test_seg_config_num_queries_match(keras_cls, pt_cls, name):
    k = keras_cls()
    p = pt_cls()
    assert k.num_queries == p.num_queries


# ===================================================================
# 2. TrainConfig parity tests
# ===================================================================

def test_train_config_defaults_lr():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.lr == p.lr


def test_train_config_defaults_batch_size():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.batch_size == p.batch_size


def test_train_config_defaults_epochs():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.epochs == p.epochs


def test_train_config_defaults_ema_decay():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.ema_decay == p.ema_decay


def test_train_config_defaults_weight_decay():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.weight_decay == p.weight_decay


def test_train_config_defaults_early_stopping():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.early_stopping == p.early_stopping


def test_train_config_defaults_dataset_file():
    k = KerasTrainConfig()
    # Keras default is 'coco_json' (backend-agnostic); PT uses 'roboflow'.
    # Both accept COCO-format JSON, just different default labels.
    assert k.dataset_file == "coco_json"


def test_train_config_defaults_tensorboard():
    k = KerasTrainConfig()
    # Keras default is False (no torch dependency); PT defaults to True.
    assert k.tensorboard is False


def test_train_config_defaults_wandb():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.wandb == p.wandb


def test_train_config_defaults_use_ema():
    k = KerasTrainConfig()
    p = PTTrainConfig(dataset_dir="/tmp")
    assert k.use_ema == p.use_ema


def test_train_config_custom_values():
    k = KerasTrainConfig(lr=0.001, epochs=50, batch_size=32)
    assert k.lr == 0.001
    assert k.epochs == 50
    assert k.batch_size == 32


def test_seg_train_config_defaults():
    k = KerasSegTrainConfig()
    p = PTSegTrainConfig(dataset_dir="/tmp")
    assert k.mask_ce_loss_coef == p.mask_ce_loss_coef
    assert k.mask_dice_loss_coef == p.mask_dice_loss_coef
    assert k.segmentation_head is True


# ===================================================================
# 3. COCO classes parity
# ===================================================================

def test_coco_classes_match():
    assert KerasCOCO == PTCOCO


def test_coco_classes_length():
    assert len(KerasCOCO) == 80


def test_coco_classes_first_is_person():
    assert KerasCOCO[1] == "person"


# ===================================================================
# 4. Variant class structure tests
# ===================================================================

KERAS_VARIANT_CLASSES = [
    (KerasRFDETRBase, "rfdetr-base"),
    (KerasRFDETRNano, "rfdetr-nano"),
    (KerasRFDETRSmall, "rfdetr-small"),
    (KerasRFDETRMedium, "rfdetr-medium"),
    (KerasRFDETRLarge, "rfdetr-large"),
    (KerasRFDETRXLarge, "rfdetr-xlarge"),
    (KerasRFDETR2XLarge, "rfdetr-2xlarge"),
]

PT_VARIANT_CLASSES = [
    (PTRFDETRBase, "rfdetr-base"),
    (PTRFDETRNano, "rfdetr-nano"),
    (PTRFDETRSmall, "rfdetr-small"),
    (PTRFDETRMedium, "rfdetr-medium"),
    (PTRFDETRLarge, "rfdetr-large"),
]

KERAS_SEG_VARIANT_CLASSES = [
    (KerasRFDETRSegPreview, "rfdetr-seg-preview"),
    (KerasRFDETRSegNano, "rfdetr-seg-nano"),
    (KerasRFDETRSegSmall, "rfdetr-seg-small"),
    (KerasRFDETRSegMedium, "rfdetr-seg-medium"),
    (KerasRFDETRSegLarge, "rfdetr-seg-large"),
    (KerasRFDETRSegXLarge, "rfdetr-seg-xlarge"),
    (KerasRFDETRSeg2XLarge, "rfdetr-seg-2xlarge"),
]

PT_SEG_VARIANT_CLASSES = [
    (PTRFDETRSegPreview, "rfdetr-seg-preview"),
    (PTRFDETRSegNano, "rfdetr-seg-nano"),
    (PTRFDETRSegSmall, "rfdetr-seg-small"),
    (PTRFDETRSegMedium, "rfdetr-seg-medium"),
    (PTRFDETRSegLarge, "rfdetr-seg-large"),
    (PTRFDETRSegXLarge, "rfdetr-seg-xlarge"),
    (PTRFDETRSeg2XLarge, "rfdetr-seg-2xlarge"),
]


@pytest.mark.parametrize("cls,expected_size", KERAS_VARIANT_CLASSES)
def test_keras_variant_size(cls, expected_size):
    assert cls.size == expected_size


@pytest.mark.parametrize("cls,expected_size", PT_VARIANT_CLASSES)
def test_pt_variant_size(cls, expected_size):
    assert cls.size == expected_size


@pytest.mark.parametrize("cls,expected_size", KERAS_SEG_VARIANT_CLASSES)
def test_keras_seg_variant_size(cls, expected_size):
    assert cls.size == expected_size


@pytest.mark.parametrize("cls,expected_size", PT_SEG_VARIANT_CLASSES)
def test_pt_seg_variant_size(cls, expected_size):
    assert cls.size == expected_size


# Check that the sizes match between Keras and PyTorch for shared variants
SHARED_SIZE_PAIRS = [
    (KerasRFDETRBase, PTRFDETRBase),
    (KerasRFDETRNano, PTRFDETRNano),
    (KerasRFDETRSmall, PTRFDETRSmall),
    (KerasRFDETRMedium, PTRFDETRMedium),
    (KerasRFDETRLarge, PTRFDETRLarge),
    (KerasRFDETRSegPreview, PTRFDETRSegPreview),
    (KerasRFDETRSegNano, PTRFDETRSegNano),
    (KerasRFDETRSegSmall, PTRFDETRSegSmall),
    (KerasRFDETRSegMedium, PTRFDETRSegMedium),
    (KerasRFDETRSegLarge, PTRFDETRSegLarge),
    (KerasRFDETRSegXLarge, PTRFDETRSegXLarge),
    (KerasRFDETRSeg2XLarge, PTRFDETRSeg2XLarge),
]


@pytest.mark.parametrize("keras_cls,pt_cls", SHARED_SIZE_PAIRS)
def test_size_attribute_parity(keras_cls, pt_cls):
    assert keras_cls.size == pt_cls.size


# ===================================================================
# 5. Variant registry tests
# ===================================================================

def test_variant_registry_has_all_detection_keys():
    for name in ["RFDETRBase", "RFDETRNano", "RFDETRSmall", "RFDETRMedium",
                  "RFDETRLarge", "RFDETRXLarge", "RFDETR2XLarge"]:
        assert name in KerasVariantRegistry


def test_variant_registry_has_all_seg_keys():
    for name in ["RFDETRSegPreview", "RFDETRSegNano", "RFDETRSegSmall",
                  "RFDETRSegMedium", "RFDETRSegLarge", "RFDETRSegXLarge",
                  "RFDETRSeg2XLarge"]:
        assert name in KerasVariantRegistry


def test_variant_registry_count():
    assert len(KerasVariantRegistry) == 14


# ===================================================================
# 6. RFDETR base class API tests (with mocked model)
# ===================================================================

def test_base_class_has_train_method(mock_keras_model):
    model = KerasRFDETRBase()
    assert hasattr(model, "train")
    assert callable(model.train)


def test_base_class_has_train_from_config(mock_keras_model):
    model = KerasRFDETRBase()
    assert hasattr(model, "train_from_config")
    assert callable(model.train_from_config)


def test_base_class_has_predict(mock_keras_model):
    model = KerasRFDETRBase()
    assert hasattr(model, "predict")


def test_base_class_has_request_early_stop(mock_keras_model):
    model = KerasRFDETRBase()
    assert hasattr(model, "request_early_stop")


def test_base_class_has_callbacks(mock_keras_model):
    model = KerasRFDETRBase()
    assert isinstance(model.callbacks, defaultdict)


def test_base_class_stop_early_default(mock_keras_model):
    model = KerasRFDETRBase()
    assert model.stop_early is False


def test_request_early_stop_sets_flag(mock_keras_model):
    model = KerasRFDETRBase()
    model.request_early_stop()
    assert model.stop_early is True


# ===================================================================
# 7. PyTorch RFDETR base class API match
# ===================================================================

def test_pt_base_has_train_method(mock_pt_model):
    model = PTRFDETRBase()
    assert hasattr(model, "train")


def test_pt_base_has_train_from_config(mock_pt_model):
    model = PTRFDETRBase()
    assert hasattr(model, "train_from_config")


def test_pt_base_has_request_early_stop(mock_pt_model):
    # The PyTorch model has request_early_stop on the Model inner class
    # The detr wrapper mirrors it via stop_early
    model = PTRFDETRBase()
    assert hasattr(model.model, "request_early_stop")


def test_pt_base_has_callbacks(mock_pt_model):
    model = PTRFDETRBase()
    assert isinstance(model.callbacks, defaultdict)


# Both Keras and PT expose the same set of public methods
def test_api_method_parity(mock_keras_model, mock_pt_model):
    k = KerasRFDETRBase()
    p = PTRFDETRBase()
    methods = ["train", "predict", "get_model_config", "get_train_config"]
    for m in methods:
        assert hasattr(k, m), f"Keras missing {m}"
        assert hasattr(p, m), f"PyTorch missing {m}"


# ===================================================================
# 8. get_model_config return type tests
# ===================================================================

def test_get_model_config_returns_correct_type(mock_keras_model):
    model = KerasRFDETRBase()
    cfg = model.get_model_config()
    assert isinstance(cfg, KerasBaseConfig)


def test_get_model_config_nano(mock_keras_model):
    model = KerasRFDETRNano()
    cfg = model.get_model_config()
    assert isinstance(cfg, KerasNanoConfig)


def test_get_model_config_kwargs_forwarded(mock_keras_model):
    model = KerasRFDETRBase()
    cfg = model.get_model_config(resolution=800)
    assert cfg.resolution == 800


# ===================================================================
# 9. get_train_config return type tests
# ===================================================================

def test_get_train_config_returns_correct_type(mock_keras_model):
    model = KerasRFDETRBase()
    cfg = model.get_train_config(dataset_dir="/tmp")
    assert isinstance(cfg, KerasTrainConfig)


def test_get_train_config_seg_variant(mock_keras_model):
    model = KerasRFDETRSegPreview()
    cfg = model.get_train_config(dataset_dir="/tmp")
    assert isinstance(cfg, KerasSegTrainConfig)


def test_get_train_config_kwargs(mock_keras_model):
    model = KerasRFDETRBase()
    cfg = model.get_train_config(lr=0.01, epochs=5, dataset_dir="/tmp")
    assert cfg.lr == 0.01
    assert cfg.epochs == 5


# PyTorch side
def test_pt_get_train_config_returns_correct_type(mock_pt_model):
    model = PTRFDETRBase()
    cfg = model.get_train_config(dataset_dir="/tmp")
    assert isinstance(cfg, PTTrainConfig)


def test_pt_get_train_config_seg(mock_pt_model):
    model = PTRFDETRSegPreview()
    cfg = model.get_train_config(dataset_dir="/tmp")
    assert isinstance(cfg, PTSegTrainConfig)


# ===================================================================
# 10. class_names property tests
# ===================================================================

def test_class_names_default_coco(mock_keras_model):
    _, mock_model = mock_keras_model
    mock_model.class_names = None
    model = KerasRFDETRBase()
    assert model.class_names == KerasCOCO


def test_class_names_custom(mock_keras_model):
    _, mock_model = mock_keras_model
    mock_model.class_names = ["cat", "dog"]
    model = KerasRFDETRBase()
    assert model.class_names == {1: "cat", 2: "dog"}


def test_pt_class_names_default_coco(mock_pt_model):
    _, mock_model = mock_pt_model
    mock_model.class_names = None
    model = PTRFDETRBase()
    assert model.class_names == PTCOCO


def test_pt_class_names_custom(mock_pt_model):
    _, mock_model = mock_pt_model
    mock_model.class_names = ["cat", "dog"]
    model = PTRFDETRBase()
    assert model.class_names == {1: "cat", 2: "dog"}


# ===================================================================
# 11. resolution property tests
# ===================================================================

def test_resolution_base(mock_keras_model):
    model = KerasRFDETRBase()
    assert model.resolution == 560


def test_resolution_nano(mock_keras_model):
    model = KerasRFDETRNano()
    assert model.resolution == 384


def test_resolution_large(mock_keras_model):
    model = KerasRFDETRLarge()
    assert model.resolution == 704


# ===================================================================
# 12. Callback wiring tests
# ===================================================================

def test_callback_append(mock_keras_model):
    model = KerasRFDETRBase()
    data_log = []
    model.callbacks["on_fit_epoch_end"].append(lambda d: data_log.append(d))
    assert len(model.callbacks["on_fit_epoch_end"]) == 1


def test_callback_fires_on_epoch_end(mock_keras_model):
    """Verify that user-registered callbacks are invoked during training."""
    model = KerasRFDETRBase()
    history = []
    model.callbacks["on_fit_epoch_end"].append(lambda d: history.append(d))

    # Prepare a dummy dataset
    tmpdir = _make_dataset_dir(num_classes=2)
    try:
        # Patch train_one_epoch to short-circuit
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.5},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(dataset_dir=tmpdir, epochs=2, batch_size=1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # At minimum, the user callback should be invoked once per epoch
    # plus the MetricsPlotSink callback
    assert len(history) >= 2


def test_on_train_end_callback(mock_keras_model):
    model = KerasRFDETRBase()
    end_called = []
    model.callbacks["on_train_end"].append(lambda: end_called.append(True))

    tmpdir = _make_dataset_dir(num_classes=2)
    try:
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.1},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(dataset_dir=tmpdir, epochs=1, batch_size=1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert len(end_called) >= 1


# ===================================================================
# 13. train_from_config annotation reading
# ===================================================================

def test_train_reads_coco_annotations(mock_keras_model):
    """train_from_config should parse annotations to get num_classes."""
    model = KerasRFDETRBase()
    tmpdir = _make_dataset_dir(num_classes=3)
    try:
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.1},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(dataset_dir=tmpdir, epochs=1, batch_size=1)
        _, mock_model = mock_keras_model
        # num_classes should have been changed to 3
        mock_model.reinitialize_detection_head.assert_called_once_with(3)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_train_invalid_dataset_file(mock_keras_model):
    model = KerasRFDETRBase()
    with pytest.raises(ValueError, match="Invalid dataset_file"):
        config = KerasTrainConfig(dataset_file="unknown", dataset_dir="/tmp")
        model.train_from_config(config)


# ===================================================================
# 14. EMA utility tests
# ===================================================================

def test_model_ema_decay():
    ema = KerasModelEma(MagicMock(weights=[]), decay=0.99, tau=0)
    assert ema._get_decay() == 0.99


def test_model_ema_decay_with_tau():
    ema = KerasModelEma(MagicMock(weights=[]), decay=0.99, tau=100)
    # First update: decay * (1 - exp(-1/100)) ≈ 0.99 * 0.00995...
    d = ema._get_decay()
    expected = 0.99 * (1 - math.exp(-1 / 100))
    assert abs(d - expected) < 1e-6


def test_model_ema_updates_counter():
    ema = KerasModelEma(MagicMock(weights=[]), decay=0.99, tau=0)
    assert ema.updates == 1


# ===================================================================
# 15. BestMetricHolder tests
# ===================================================================

def test_best_metric_holder_no_ema():
    h = KerasBestMetricHolder(use_ema=False)
    assert h.update(0.5, 0) is True
    assert h.update(0.4, 1) is False
    assert h.update(0.6, 2) is True


def test_best_metric_holder_with_ema():
    h = KerasBestMetricHolder(use_ema=True)
    h.update(0.5, 0, is_ema=False)
    h.update(0.7, 1, is_ema=True)
    s = h.summary()
    assert "all_best_res" in s
    assert "ema_best_res" in s


def test_best_metric_holder_summary_keys():
    h = KerasBestMetricHolder(use_ema=False)
    h.update(0.5, 0)
    s = h.summary()
    assert "best_res" in s
    assert "best_ep" in s


# ===================================================================
# 16. EarlyStoppingCallback tests
# ===================================================================

def test_early_stopping_no_improvement():
    mock_model = MagicMock()
    es = KerasEarlyStoppingCallback(model=mock_model, patience=2, min_delta=0.01)
    es.update({"test_coco_eval_bbox": [0.5]})
    es.update({"test_coco_eval_bbox": [0.5]})
    es.update({"test_coco_eval_bbox": [0.5]})
    # After 2 epochs with no improvement, should trigger
    assert es.counter >= 2


def test_early_stopping_with_improvement():
    mock_model = MagicMock()
    es = KerasEarlyStoppingCallback(model=mock_model, patience=3, min_delta=0.01)
    es.update({"test_coco_eval_bbox": [0.5]})
    es.update({"test_coco_eval_bbox": [0.6]})
    assert es.counter == 0


def test_early_stopping_calls_request_early_stop():
    # Use spec to prevent MagicMock from auto-creating 'stop_training',
    # so the callback falls through to request_early_stop().
    mock_model = MagicMock(spec=['request_early_stop'])
    mock_model.request_early_stop = MagicMock()
    es = KerasEarlyStoppingCallback(model=mock_model, patience=1, min_delta=0.01)
    es.update({"test_coco_eval_bbox": [0.5]})
    es.update({"test_coco_eval_bbox": [0.5]})
    mock_model.request_early_stop.assert_called()


def test_early_stopping_seg_metric():
    mock_model = MagicMock()
    es = KerasEarlyStoppingCallback(
        model=mock_model, patience=2, min_delta=0.0, segmentation_head=True
    )
    es.update({"test_coco_eval_masks": [0.3]})
    assert es.best_map == 0.3


# ===================================================================
# 17. MetricsPlotSink tests
# ===================================================================

def test_metrics_plot_sink_update():
    sink = KerasPlotSink(output_dir="/tmp")
    sink.update({"epoch": 0, "train_loss": 1.0})
    assert len(sink.history) == 1


def test_metrics_plot_sink_save(tmp_path):
    sink = KerasPlotSink(output_dir=str(tmp_path))
    sink.update({"epoch": 0, "train_loss": 1.0})
    sink.save()
    assert (tmp_path / "metrics_plot.png").exists()


# ===================================================================
# 18. LR schedule parity
# ===================================================================

def test_lr_lambda_step_schedule():
    lr_fn = keras_build_lr_lambda(
        num_training_steps_per_epoch=100,
        epochs=10,
        warmup_epochs=0,
        lr_scheduler="step",
        lr_drop=5,
    )
    assert lr_fn(0) == 1.0
    assert lr_fn(499) == 1.0
    assert lr_fn(500) == pytest.approx(0.1)


def test_lr_lambda_warmup():
    lr_fn = keras_build_lr_lambda(
        num_training_steps_per_epoch=100,
        epochs=10,
        warmup_epochs=1,
        lr_scheduler="step",
        lr_drop=10,
    )
    assert lr_fn(0) == 0.0
    assert lr_fn(50) == pytest.approx(0.5)
    assert lr_fn(100) == pytest.approx(1.0)


def test_lr_lambda_cosine():
    lr_fn = keras_build_lr_lambda(
        num_training_steps_per_epoch=100,
        epochs=10,
        warmup_epochs=0,
        lr_scheduler="cosine",
    )
    # At step 0, cosine should return 1.0
    assert lr_fn(0) == pytest.approx(1.0, abs=0.01)
    # At the end, should approach lr_min_factor (0.0 by default)
    assert lr_fn(999) == pytest.approx(0.0, abs=0.01)


# ===================================================================
# 19. _COCODataLoader tests
# ===================================================================

def test_coco_data_loader_creates():
    tmpdir = _make_dataset_dir(num_classes=2)
    try:
        loader = _COCODataLoader(
            ann_file=os.path.join(tmpdir, "train", "_annotations.coco.json"),
            img_dir=os.path.join(tmpdir, "train"),
            batch_size=2,
            resolution=64,
        )
        assert len(loader) >= 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_coco_data_loader_yields_batches():
    tmpdir = _make_dataset_dir(num_classes=2)
    try:
        loader = _COCODataLoader(
            ann_file=os.path.join(tmpdir, "train", "_annotations.coco.json"),
            img_dir=os.path.join(tmpdir, "train"),
            batch_size=2,
            resolution=64,
        )
        batch = next(iter(loader))
        images, targets = batch
        assert images.ndim == 4
        assert images.shape[-1] == 3
        assert images.shape[1] == 64
        assert isinstance(targets, list)
        assert "labels" in targets[0]
        assert "boxes" in targets[0]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_coco_data_loader_target_boxes_normalised():
    tmpdir = _make_dataset_dir(num_classes=2)
    try:
        loader = _COCODataLoader(
            ann_file=os.path.join(tmpdir, "train", "_annotations.coco.json"),
            img_dir=os.path.join(tmpdir, "train"),
            batch_size=4,
            resolution=64,
        )
        for images, targets in loader:
            for t in targets:
                if len(t["boxes"]) > 0:
                    assert np.all(t["boxes"] >= 0)
                    assert np.all(t["boxes"] <= 1)
            break
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# 20. Full training integration test (mocked)
# ===================================================================

def test_full_training_loop_mocked(mock_keras_model):
    """Integration: Run a full mock training loop with callbacks."""
    model = KerasRFDETRBase()
    history = []
    model.callbacks["on_fit_epoch_end"].append(lambda d: history.append(d))

    tmpdir = _make_dataset_dir(num_classes=2)
    try:
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.42, "loss_ce": 0.1, "loss_bbox": 0.2},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(
                dataset_dir=tmpdir,
                epochs=3,
                batch_size=2,
                lr=1e-4,
                tensorboard=False,
                wandb=False,
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert len(history) >= 3
    assert "epoch" in history[0]
    assert history[-1]["epoch"] == 2  # 0-indexed


def test_training_creates_checkpoint(mock_keras_model):
    model = KerasRFDETRBase()
    tmpdir = _make_dataset_dir(num_classes=2)
    out_dir = tempfile.mkdtemp(prefix="rfdetr_out_")
    try:
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.1},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(
                dataset_dir=tmpdir,
                epochs=1,
                batch_size=1,
                output_dir=out_dir,
                tensorboard=False,
                wandb=False,
            )
        # Should create log.txt
        assert os.path.isfile(os.path.join(out_dir, "log.txt"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_early_stop_terminates_loop(mock_keras_model):
    """If stop_early is set, the training loop should exit."""
    model = KerasRFDETRBase()
    history = []
    model.callbacks["on_fit_epoch_end"].append(lambda d: history.append(d))

    # Set stop_early after epoch 0 via callback
    def setter(d):
        if d.get("epoch", 0) >= 0:
            model.stop_early = True

    model.callbacks["on_fit_epoch_end"].append(setter)

    tmpdir = _make_dataset_dir(num_classes=2)
    try:
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.1},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(
                dataset_dir=tmpdir, epochs=10, batch_size=1,
                tensorboard=False, wandb=False,
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Should stop after 1 epoch despite epochs=10
    assert len(history) <= 3  # epoch 0 + max 2 callbacks


# ===================================================================
# 21. PyTorch vs Keras API signature comparison
# ===================================================================

def test_train_accepts_dataset_dir_kwarg(mock_keras_model):
    """Both implementations accept dataset_dir as kwarg."""
    model = KerasRFDETRBase()
    tmpdir = _make_dataset_dir()
    try:
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.1},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(dataset_dir=tmpdir, epochs=1, batch_size=1,
                        tensorboard=False, wandb=False)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_train_accepts_lr_kwarg(mock_keras_model):
    model = KerasRFDETRBase()
    tmpdir = _make_dataset_dir()
    try:
        with patch(
            "paz.models.detection.dino_v2_object_detection.engine.train_one_epoch",
            return_value={"loss": 0.1},
        ), patch(
            "paz.models.detection.dino_v2_object_detection.detr.build_criterion_from_config",
            return_value=(MagicMock(), MagicMock()),
        ):
            model.train(dataset_dir=tmpdir, epochs=1, batch_size=1, lr=0.001,
                        tensorboard=False, wandb=False)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# 22. Predict API parity
# ===================================================================

def test_predict_accepts_list(mock_keras_model):
    model = KerasRFDETRBase()
    _, mock_model = mock_keras_model
    mock_model.predict = MagicMock(return_value=[{"boxes": [], "scores": [], "labels": []}])
    imgs = [np.random.rand(64, 64, 3).astype("float32")]
    result = model.predict(imgs, threshold=0.5)
    mock_model.predict.assert_called_once()


def test_predict_accepts_uint8(mock_keras_model):
    model = KerasRFDETRBase()
    _, mock_model = mock_keras_model
    mock_model.predict = MagicMock(return_value=[{"boxes": [], "scores": [], "labels": []}])
    imgs = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
    result = model.predict(imgs, threshold=0.5)
    # Should have been converted to float internally
    call_args = mock_model.predict.call_args
    assert call_args[0][0].dtype == np.float32


def test_predict_accepts_3d_array(mock_keras_model):
    model = KerasRFDETRBase()
    _, mock_model = mock_keras_model
    mock_model.predict = MagicMock(return_value=[{"boxes": [], "scores": [], "labels": []}])
    img = np.random.rand(64, 64, 3).astype("float32")
    model.predict(img, threshold=0.5)
    call_args = mock_model.predict.call_args
    assert call_args[0][0].ndim == 4


# ===================================================================
# 23. Configuration field coverage
# ===================================================================

def test_train_config_has_all_pt_fields():
    """Ensure Keras TrainConfig has at least all fields that PyTorch has."""
    pt_fields = set(PTTrainConfig.model_fields.keys())
    keras_fields = {f.name for f in fields(KerasTrainConfig)}
    # Allow Keras to have extra fields but must have all PT fields
    missing = pt_fields - keras_fields
    # Filter out fields that may differ by design:
    # - square_resize_div_64 / do_random_resize_via_padding: Keras-specific padding
    # - num_select: belongs in ModelConfig, not TrainConfig
    # - resume: PyTorch-specific checkpoint resume path
    allowed_missing = {
        "square_resize_div_64", "do_random_resize_via_padding",
        "num_select", "resume",
    }
    actual_missing = missing - allowed_missing
    assert actual_missing == set(), f"Keras TrainConfig missing: {actual_missing}"


def test_model_config_shared_fields():
    """Core fields should exist in both Keras and PyTorch ModelConfig."""
    core_fields = [
        "encoder", "hidden_dim", "dec_layers", "num_classes", "resolution",
        "patch_size", "num_windows", "sa_nheads", "ca_nheads", "dec_n_points",
        "group_detr", "segmentation_head",
    ]
    keras_fields = {f.name for f in fields(KerasModelConfig)}
    pt_fields = set(PTModelConfig.model_fields.keys())
    for field_name in core_fields:
        assert field_name in keras_fields, f"Keras ModelConfig missing: {field_name}"
        assert field_name in pt_fields, f"PT ModelConfig missing: {field_name}"


# ===================================================================
# 24. Seg variant get_train_config type
# ===================================================================

SEG_KERAS_CLASSES = [
    KerasRFDETRSegPreview, KerasRFDETRSegNano, KerasRFDETRSegSmall,
    KerasRFDETRSegMedium, KerasRFDETRSegLarge, KerasRFDETRSegXLarge,
    KerasRFDETRSeg2XLarge,
]


@pytest.mark.parametrize("cls", SEG_KERAS_CLASSES)
def test_seg_variant_returns_seg_train_config(cls, mock_keras_model):
    model = cls()
    cfg = model.get_train_config(dataset_dir="/tmp")
    assert isinstance(cfg, KerasSegTrainConfig)


SEG_PT_CLASSES = [
    PTRFDETRSegPreview, PTRFDETRSegNano, PTRFDETRSegSmall,
    PTRFDETRSegMedium, PTRFDETRSegLarge, PTRFDETRSegXLarge,
    PTRFDETRSeg2XLarge,
]


@pytest.mark.parametrize("cls", SEG_PT_CLASSES)
def test_pt_seg_variant_returns_seg_train_config(cls, mock_pt_model):
    model = cls()
    cfg = model.get_train_config(dataset_dir="/tmp")
    assert isinstance(cfg, PTSegTrainConfig)


# ===================================================================
# 25. Drop scheduler parity
# ===================================================================

def test_drop_scheduler_standard():
    from paz.models.detection.dino_v2_object_detection.utils.drop_scheduler import (
        drop_scheduler as keras_drop,
    )
    from rfdetr.util.drop_scheduler import drop_scheduler as pt_drop

    k = keras_drop(0.1, 5, 10)
    p = pt_drop(0.1, 5, 10)
    np.testing.assert_array_almost_equal(k, p)


def test_drop_scheduler_early_constant():
    from paz.models.detection.dino_v2_object_detection.utils.drop_scheduler import (
        drop_scheduler as keras_drop,
    )
    from rfdetr.util.drop_scheduler import drop_scheduler as pt_drop

    k = keras_drop(0.2, 10, 5, cutoff_epoch=5, mode='early', schedule='constant')
    p = pt_drop(0.2, 10, 5, cutoff_epoch=5, mode='early', schedule='constant')
    np.testing.assert_array_almost_equal(k, p)


def test_drop_scheduler_early_linear():
    from paz.models.detection.dino_v2_object_detection.utils.drop_scheduler import (
        drop_scheduler as keras_drop,
    )
    from rfdetr.util.drop_scheduler import drop_scheduler as pt_drop

    k = keras_drop(0.3, 10, 5, cutoff_epoch=3, mode='early', schedule='linear')
    p = pt_drop(0.3, 10, 5, cutoff_epoch=3, mode='early', schedule='linear')
    np.testing.assert_array_almost_equal(k, p)


# ===================================================================
# 26. means / stds parity
# ===================================================================

def test_means_parity(mock_keras_model, mock_pt_model):
    k = KerasRFDETRBase()
    p = PTRFDETRBase()
    np.testing.assert_array_almost_equal(np.array(k.means), np.array(p.means))


def test_stds_parity(mock_keras_model, mock_pt_model):
    k = KerasRFDETRBase()
    p = PTRFDETRBase()
    np.testing.assert_array_almost_equal(np.array(k.stds), np.array(p.stds))


# ===================================================================
# 27. MetricsTensorBoardSink tests
# ===================================================================

def test_tb_sink_no_crash_without_tensorboard():
    """Should gracefully handle missing tensorboard."""
    sink = KerasTBSink(output_dir="/tmp")
    sink.update({"epoch": 0, "train_loss": 1.0})
    sink.close()


# ===================================================================
# 28. MetricsWandBSink tests
# ===================================================================

def test_wandb_sink_no_crash_without_wandb():
    """Should gracefully handle missing wandb."""
    sink = KerasWBSink(output_dir="/tmp", project="test", run="test")
    sink.update({"epoch": 0})
    sink.close()


# ===================================================================
# 29. Dataclass serialisation round-trip
# ===================================================================

def test_train_config_round_trip():
    cfg = KerasTrainConfig(lr=0.005, epochs=20, batch_size=8, dataset_dir="/data")
    d = asdict(cfg)
    cfg2 = KerasTrainConfig(**d)
    assert cfg2.lr == 0.005
    assert cfg2.epochs == 20


def test_model_config_round_trip():
    cfg = KerasBaseConfig(resolution=800)
    d = asdict(cfg)
    cfg2 = KerasBaseConfig(**d)
    assert cfg2.resolution == 800


# ===================================================================
# 30. Edge cases
# ===================================================================

def test_predict_empty_batch(mock_keras_model):
    model = KerasRFDETRBase()
    _, mock_model = mock_keras_model
    mock_model.predict = MagicMock(return_value=[])
    imgs = np.random.rand(0, 64, 64, 3).astype("float32")
    # Should not crash - implementation may raise or return empty
    try:
        model.predict(imgs)
    except (ValueError, IndexError):
        pass  # acceptable


def test_callback_defaultdict_unknown_key(mock_keras_model):
    model = KerasRFDETRBase()
    # Accessing an unknown key should return empty list
    assert model.callbacks["nonexistent_event"] == []


def test_train_config_default_output_dir():
    cfg = KerasTrainConfig()
    assert cfg.output_dir == "output"


def test_train_config_default_dataset_file():
    cfg = KerasTrainConfig()
    assert cfg.dataset_file == "coco_json"
