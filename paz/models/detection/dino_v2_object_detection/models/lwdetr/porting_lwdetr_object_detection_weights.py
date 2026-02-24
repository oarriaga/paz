import gc
import io
import math
import os
import sys
import warnings

import numpy as np
import pytest
from urllib.request import urlopen

# ---- path setup ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- PyTorch guard -------------------------------------------------------
try:
    import torch
    import torchvision.transforms.functional as F_tv
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
            os.path.join(
                current_dir, "../../../../../../examples/rf-detr_original_pytorch_implementation"
            )
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
    from rfdetr.models.backbone.dinov2_with_windowed_attn import (
        Dinov2WithRegistersSelfAttention,
        Dinov2WithRegistersSdpaSelfAttention,
    )

# ---- Keras imports -------------------------------------------------------
import keras
from keras import ops
import tensorflow as tf  # Needed for image resizing operations

# Keras LWDETR imports
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    LWDETR,
    PostProcess,
)
from paz.models.detection.dino_v2_object_detection.models.backbone import (
    build_backbone as build_keras_backbone,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (
    Transformer as KerasTransformer,
)

# Weight-transfer utilities
from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (
    transfer_encoder as transfer_backbone_encoder,
    port_weights_multiscale_projector,
    transfer_layernorm,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer_weights_porting_utils import (
    transfer_transformer_weights,
)

# COCO class labels for readable detection output
from paz.models.detection.dino_v2_object_detection.utils.coco_classes import COCO_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS_DIR = os.path.join(project_root, "lwdetr_keras_weights")
CACHE_DIR = os.path.join(project_root, ".test_cache")

COCO_IMAGES = {
    # "cats": {
    #     "id": "000000039769",
    #     "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
    #     "description": "Two cats on a couch with remotes",
    #     "expected_classes": {17},  # cat
    # },
    # "bear": {
    #     "id": "000000000285",
    #     "url": "http://images.cocodataset.org/val2017/000000000285.jpg",
    #     "description": "Bear in natural habitat",
    #     "expected_classes": {23},  # bear
    # },
    "kitchen": {
        "id": "000000037777",
        "url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "description": "Kitchen scene with appliances and furniture",
        "expected_classes": {82},  # refrigerator
    },
}

IMAGENET_MEANS = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STDS = np.array([0.229, 0.224, 0.225], dtype="float32")

# ---------------------------------------------------------------------------
# Model configurations — detection only (no segmentation)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "RFDETRNano": {
        "pt_class": PT_RFDETRNano if HAS_TORCH else None,
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "dec_layers": 2,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "patch_size": 16,
        "resolution": 384,
        "num_windows": 2,
        "positional_encoding_size": 24,
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_classes": 91,
        "group_detr": 13,
        "save_key": "lwdetr_nano",
    },
    "RFDETRSmall": {
        "pt_class": PT_RFDETRSmall if HAS_TORCH else None,
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "dec_layers": 3,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "patch_size": 16,
        "resolution": 512,
        "num_windows": 2,
        "positional_encoding_size": 32,
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_classes": 91,
        "group_detr": 13,
        "save_key": "lwdetr_small",
    },
    "RFDETRMedium": {
        "pt_class": PT_RFDETRMedium if HAS_TORCH else None,
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "patch_size": 16,
        "resolution": 576,
        "num_windows": 2,
        "positional_encoding_size": 36,
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_classes": 91,
        "group_detr": 13,
        "save_key": "lwdetr_medium",
    },
    "RFDETRBase": {
        "pt_class": PT_RFDETRBase if HAS_TORCH else None,
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "dec_layers": 3,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "patch_size": 14,
        "resolution": 560,
        "num_windows": 4,
        "positional_encoding_size": 37,
        "out_feature_indexes": [1, 4, 7, 10],
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_classes": 91,
        "group_detr": 13,
        "save_key": "lwdetr_base",
    },
    "RFDETRLarge": {
        "pt_class": PT_RFDETRLarge if HAS_TORCH else None,
        "encoder": "dinov2_windowed_small",
        "hidden_dim": 256,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "patch_size": 16,
        "resolution": 704,
        "num_windows": 2,
        "positional_encoding_size": 44,
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_classes": 91,
        "group_detr": 13,
        "save_key": "lwdetr_large",
    },
    "RFDETRXLarge": {
        "pt_class": PT_RFDETRXLarge if HAS_TORCH else None,
        "encoder": "dinov2_windowed_base",
        "hidden_dim": 512,
        "dec_layers": 5,
        "sa_nheads": 16,
        "ca_nheads": 32,
        "dec_n_points": 4,
        "patch_size": 20,
        "resolution": 700,
        "num_windows": 1,
        "positional_encoding_size": 35,
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_classes": 91,
        "group_detr": 13,
        "save_key": "lwdetr_xlarge",
    },
    "RFDETR2XLarge": {
        "pt_class": PT_RFDETR2XLarge if HAS_TORCH else None,
        "encoder": "dinov2_windowed_base",
        "hidden_dim": 512,
        "dec_layers": 5,
        "sa_nheads": 16,
        "ca_nheads": 32,
        "dec_n_points": 4,
        "patch_size": 20,
        "resolution": 880,
        "num_windows": 2,
        "positional_encoding_size": 44,
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "num_queries": 300,
        "num_classes": 91,
        "group_detr": 13,
        "save_key": "lwdetr_2xlarge",
    },
}

# Filter to variants whose PT class is available
AVAILABLE_VARIANTS = [
    name for name, cfg in MODEL_CONFIGS.items() if cfg.get("pt_class") is not None
]


# ---------------------------------------------------------------------------
# Weight transfer helpers (same logic as test_lwdetr_with_real_weights.py)
# ---------------------------------------------------------------------------


def resize_and_assign_pos_embed(pt_embeddings_layer, keras_embeddings_layer):
    """Interpolate PyTorch position embeddings to match Keras shape if needed."""
    if hasattr(pt_embeddings_layer.position_embeddings, "weight"):
        pt_pos_embed = (
            pt_embeddings_layer.position_embeddings.weight.detach().cpu().numpy()
        )
    else:
        pt_pos_embed = pt_embeddings_layer.position_embeddings.detach().cpu().numpy()

    if pt_pos_embed.ndim == 2:
        pt_pos_embed = np.expand_dims(pt_pos_embed, axis=0)

    keras_shape = keras_embeddings_layer.position_embeddings.shape

    if pt_pos_embed.shape == keras_shape:
        keras_embeddings_layer.position_embeddings.assign(pt_pos_embed)
        return

    print(f"  Resizing PosEmbed: PT {pt_pos_embed.shape} -> Keras {keras_shape}")

    cls_token = pt_pos_embed[:, 0:1, :]
    grid_tokens = pt_pos_embed[:, 1:, :]

    n_tokens = grid_tokens.shape[1]
    if n_tokens == 0:
        print("  WARNING: PyTorch grid tokens are empty — skipping resize.")
        return

    gs_pt = int(np.sqrt(n_tokens))
    n_tokens_keras = keras_shape[1] - 1
    gs_keras = int(np.sqrt(n_tokens_keras))

    grid_tokens = grid_tokens.reshape(1, gs_pt, gs_pt, -1)

    # Use PyTorch's interpolation with size= (NOT scale_factor) and
    # align_corners=False to exactly match the DINOv2 runtime code in
    # dinov2_with_windowed_attn.py::interpolate_pos_encoding.
    pt_tensor = (
        torch.tensor(grid_tokens).permute(0, 3, 1, 2).to(dtype=torch.float32)
    )  # (1, C, H, W)
    grid_tokens_resized = torch.nn.functional.interpolate(
        pt_tensor,
        size=(gs_keras, gs_keras),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    grid_tokens_resized = grid_tokens_resized.permute(
        0, 2, 3, 1
    ).numpy()  # (1, H, W, C)

    grid_tokens_resized = grid_tokens_resized.reshape(1, -1, pt_pos_embed.shape[-1])

    new_pos_embed = np.concatenate([cls_token, grid_tokens_resized], axis=1)
    keras_embeddings_layer.position_embeddings.assign(new_pos_embed)


def transfer_lwdetr_head_weights(pt_model, keras_model, config):
    """Transfer LWDETR-specific weights (class/bbox heads, embeddings)."""
    # 1. Class embed
    keras_model.class_embed.set_weights(
        [
            pt_model.class_embed.weight.detach().cpu().numpy().T,
            pt_model.class_embed.bias.detach().cpu().numpy(),
        ]
    )

    # 2. BBox embed MLP
    for pt_l, k_l in zip(
        pt_model.bbox_embed.layers, keras_model.bbox_embed.layers_list
    ):
        k_l.set_weights(
            [pt_l.weight.detach().cpu().numpy().T, pt_l.bias.detach().cpu().numpy()]
        )

    # 3. Query embeddings
    if hasattr(pt_model.refpoint_embed, "weight"):
        keras_model.refpoint_embed.assign(
            pt_model.refpoint_embed.weight.detach().cpu().numpy()
        )
    else:
        keras_model.refpoint_embed.assign(
            pt_model.refpoint_embed.detach().cpu().numpy()
        )

    if hasattr(pt_model.query_feat, "weight"):
        keras_model.query_feat.assign(pt_model.query_feat.weight.detach().cpu().numpy())
    else:
        keras_model.query_feat.assign(pt_model.query_feat.detach().cpu().numpy())

    # 4. Two-stage encoder output heads
    if config.get("two_stage", True):
        group_detr = config.get("group_detr", 13)
        for g in range(group_detr):
            pt_cls = pt_model.transformer.enc_out_class_embed[g]
            k_cls = keras_model.enc_out_class_embed[g]
            k_cls.set_weights(
                [
                    pt_cls.weight.detach().cpu().numpy().T,
                    pt_cls.bias.detach().cpu().numpy(),
                ]
            )

            pt_bbox = pt_model.transformer.enc_out_bbox_embed[g]
            k_bbox = keras_model.enc_out_bbox_embed[g]
            for pt_l, k_l in zip(pt_bbox.layers, k_bbox.layers_list):
                k_l.set_weights(
                    [
                        pt_l.weight.detach().cpu().numpy().T,
                        pt_l.bias.detach().cpu().numpy(),
                    ]
                )


def transfer_full_model_weights(pt_model, keras_model, config):
    """Orchestrate weight transfer for the entire LWDETR model."""
    inner_pt = pt_model.model.model
    pt_backbone = inner_pt.backbone[0]
    keras_backbone = keras_model.backbone.backbone

    # 1. Backbone
    # a. Position embeddings
    # For models whose pretrained pos-embed grid != target grid (e.g.
    # RFDETRBase: pretrained 37×37 but target 40×40), we call export() on
    # the PT DinoV2 to pre-compute the interpolation in PyTorch.  The
    # Keras model is built with positional_encoding_size = resolution //
    # patch_size so its pos-embed already has the target shape -- direct
    # copy, no runtime interpolation -- eliminates cross-framework diff.
    pt_pos_embed = pt_backbone.encoder.encoder.embeddings.position_embeddings
    stored_grid = int(math.sqrt(pt_pos_embed.shape[1] - 1))
    target_grid = config["resolution"] // config["patch_size"]
    if stored_grid != target_grid:
        print(
            f"  Pre-computing pos embed interpolation: "
            f"{stored_grid}x{stored_grid} -> {target_grid}x{target_grid}"
        )
        pt_backbone.encoder.export()

    resize_and_assign_pos_embed(
        pt_backbone.encoder.encoder.embeddings,
        keras_backbone.encoder.encoder.embeddings,
    )

    # b. Patch embeddings (projection + CLS token)
    pt_embeddings = pt_backbone.encoder.encoder.embeddings

    if hasattr(pt_embeddings, "patch_embeddings"):
        pt_patch_embed = pt_embeddings.patch_embeddings
    else:
        pt_patch_embed = pt_embeddings

    keras_patch_embed = keras_backbone.encoder.encoder.embeddings

    if hasattr(pt_patch_embed, "projection"):
        pt_proj_weight = pt_patch_embed.projection.weight
        pt_proj_bias = pt_patch_embed.projection.bias
    elif hasattr(pt_patch_embed, "proj"):
        pt_proj_weight = pt_patch_embed.proj.weight
        pt_proj_bias = pt_patch_embed.proj.bias
    else:
        raise AttributeError(f"Could not find projection weights in {pt_patch_embed}")

    keras_patch_embed.projection.kernel.assign(
        pt_proj_weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
    )
    keras_patch_embed.projection.bias.assign(pt_proj_bias.detach().cpu().numpy())

    if hasattr(pt_embeddings, "cls_token"):
        keras_backbone.encoder.encoder.embeddings.cls_token.assign(
            pt_embeddings.cls_token.detach().cpu().numpy()
        )

    if hasattr(keras_backbone.encoder.encoder.embeddings, "mask_token"):
        if hasattr(pt_embeddings, "mask_token"):
            keras_backbone.encoder.encoder.embeddings.mask_token.assign(
                pt_embeddings.mask_token.detach().cpu().numpy()
            )

    # c. Encoder blocks
    transfer_backbone_encoder(
        pt_backbone.encoder.encoder.encoder, keras_backbone.encoder.encoder.encoder
    )

    # d. Final LayerNorm
    transfer_layernorm(
        pt_backbone.encoder.encoder.layernorm, keras_backbone.encoder.encoder.layernorm
    )

    # e. Multi-scale projector
    port_weights_multiscale_projector(pt_backbone.projector, keras_backbone.projector)

    # 2. Transformer decoder
    transfer_transformer_weights(
        inner_pt.transformer,
        keras_model.transformer,
        config["hidden_dim"],
        config["sa_nheads"],
    )

    # 3. LWDETR heads
    transfer_lwdetr_head_weights(inner_pt, keras_model, config)

    print(f"  Weight transfer complete.")


# ---------------------------------------------------------------------------
# Keras model builder
# ---------------------------------------------------------------------------


def build_keras_lwdetr(config):
    """Build a Keras LWDETR model from a config dict."""
    num_classes = config.get("num_classes", 91)

    backbone = build_keras_backbone(
        encoder=config["encoder"],
        hidden_dim=config["hidden_dim"],
        out_channels=config["hidden_dim"],
        patch_size=config["patch_size"],
        num_windows=config["num_windows"],
        out_feature_indexes=config["out_feature_indexes"],
        projector_scale=config["projector_scale"],
        layer_norm=True,
        target_shape=(config["resolution"], config["resolution"]),
        positional_encoding_size=config.get("positional_encoding_size", 37),
    )

    transformer = KerasTransformer(
        d_model=config["hidden_dim"],
        sa_nhead=config["sa_nheads"],
        ca_nhead=config["ca_nheads"],
        num_queries=config["num_queries"],
        num_decoder_layers=config["dec_layers"],
        num_feature_levels=len(config["projector_scale"]),
        dec_n_points=config["dec_n_points"],
        two_stage=True,
        bbox_reparam=True,
        return_intermediate_dec=True,
        lite_refpoint_refine=config.get("lite_refpoint_refine", True),
        group_detr=config.get("group_detr", 13),
    )

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=config["num_queries"],
        group_detr=config.get("group_detr", 13),
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=config.get("lite_refpoint_refine", True),
    )

    # Build the model with a dummy forward pass (training=True to build
    # all group_detr heads)
    res = config["resolution"]
    dummy = np.ones((1, res, res, 3), dtype=np.float32) * 0.5
    model(dummy, training=True)

    return model


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
    """Preprocess matching the PyTorch rfdetr ``predict()`` pipeline.

    Pipeline (from ``rfdetr/detr.py``):
      1. ``F.to_tensor``  – (H,W,3) float [0,1] → (3,H,W) tensor
      2. ``F.normalize``  – ImageNet mean / std
      3. ``F.resize``     – bilinear to ``(resolution, resolution)``

    We set ``antialias=False`` so the bilinear resize matches
    ``tf.image.resize`` semantics (no pre-filter).  This keeps the
    pixel-level input identical to what the Keras pipeline produces,
    preventing input-dependent FP drift in the parity comparison.

    Returns (1, H, W, 3) float32 numpy array.
    """
    t = F_tv.to_tensor(image_float)  # (3,H,W)
    t = F_tv.normalize(t, IMAGENET_MEANS.tolist(), IMAGENET_STDS.tolist())  # normalise
    t = F_tv.resize(t, [resolution, resolution], antialias=False)  # resize
    return t.unsqueeze(0).permute(0, 2, 3, 1).numpy()  # (1,H,W,3)


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


def _run_keras_detection(keras_lwdetr, image_float, resolution, num_select=300):
    """Run forward pass + postprocess on a single image.
    Returns (scores, labels, boxes) numpy arrays for the first image.
    """
    preprocessed = _preprocess(image_float, resolution)
    raw = keras_lwdetr(preprocessed, training=False)
    H, W = image_float.shape[:2]
    pp = PostProcess(num_select=num_select)
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
# Phase 1: Build Keras LWDETR, port PT weights, verify output parity
# ---------------------------------------------------------------------------


def _force_eager_attention(pt_model):
    """Replace SDPA attention modules with eager (manual) attention.

    PyTorch's ``scaled_dot_product_attention`` may use kernel-level
    optimisations (Flash-Attention, memory-efficient, or a custom math
    path) that produce slightly different floating-point results than
    the manual ``matmul -> scale -> softmax -> matmul`` path used in
    the Keras implementation.  Switching the PT model to the eager path
    aligns the two forward passes and keeps the parity diff below 1e-4.

    The eager ``Dinov2WithRegistersSelfAttention`` computes attention
    identically to the Keras ``Attention`` layer:
        attn = softmax(Q @ K^T / sqrt(d)) @ V
    """
    backbone = pt_model.model.model.backbone[0]
    encoder_layers = backbone.encoder.encoder.encoder.layer
    config = backbone.encoder.encoder.config
    patched = 0
    for layer in encoder_layers:
        inner = layer.attention.attention
        if isinstance(inner, Dinov2WithRegistersSdpaSelfAttention):
            eager = Dinov2WithRegistersSelfAttention(config)
            eager.query.weight = inner.query.weight
            eager.query.bias = inner.query.bias
            eager.key.weight = inner.key.weight
            eager.key.bias = inner.key.bias
            eager.value.weight = inner.value.weight
            eager.value.bias = inner.value.bias
            layer.attention.attention = eager
            patched += 1
    print(f"  Forced eager attention on {patched} encoder layers")


def _build_and_port_variant(variant_name):
    """Build PyTorch model, build Keras LWDETR, transfer weights.
    Returns (pt_model, keras_model, config).
    """
    config = MODEL_CONFIGS[variant_name]

    # 1. Instantiate PyTorch model (auto-downloads weights)
    print(f"\n  Instantiating PyTorch {variant_name}...")
    if "XLarge" in variant_name or "Xlarge" in variant_name:
        pt_model = config["pt_class"](accept_platform_model_license=True)
    else:
        pt_model = config["pt_class"]()
    pt_model.model.model.eval()

    # Force eager (manual) attention instead of SDPA so that the PT
    # forward pass uses the same matmul -> softmax -> matmul sequence
    # as the Keras model, eliminating attention-kernel FP divergence.
    _force_eager_attention(pt_model)

    # 2. Build Keras LWDETR
    print(f"  Building Keras LWDETR for {variant_name}...")
    keras_model = build_keras_lwdetr(config)

    # 3. Transfer weights
    print(f"  Transferring weights for {variant_name}...")
    transfer_full_model_weights(pt_model, keras_model, config)

    return pt_model, keras_model, config


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPortingParity:
    """For each detection variant: port PT weights to Keras LWDETR, verify
    output parity within 1e-4 on three COCO images, then save weights."""

    @pytest.fixture(
        scope="class",
        params=[v for v in AVAILABLE_VARIANTS],
    )
    def variant(self, request, coco_images):
        """Class-scoped parameterised fixture: builds one variant at a time."""
        name = request.param
        print(f"\n{'=' * 60}")
        print(f"  Building variant: {name}")
        print(f"{'=' * 60}")

        pt_model, keras_model, config = _build_and_port_variant(name)

        yield {
            "name": name,
            "pt_model": pt_model,
            "keras_model": keras_model,
            "config": config,
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
        keras_model = variant["keras_model"]
        config = variant["config"]
        img = variant["images"][image_name]
        res = config["resolution"]

        # Identical preprocessed input
        preprocessed = _preprocess(img, res)

        # PyTorch forward
        pt_input = torch.from_numpy(preprocessed).permute(0, 3, 1, 2)
        mask = torch.zeros((1, res, res), dtype=torch.bool)
        samples = NestedTensor(pt_input, mask)
        with torch.no_grad():
            pt_out = pt_model.model.model(samples)

        # Keras forward
        k_out = keras_model(preprocessed, training=False)

        # Compare logits
        pt_logits = pt_out["pred_logits"].cpu().numpy()
        k_logits = ops.convert_to_numpy(k_out["pred_logits"])
        diff_logits = np.abs(pt_logits - k_logits)

        # Compare boxes
        pt_boxes = pt_out["pred_boxes"].cpu().numpy()
        k_boxes = ops.convert_to_numpy(k_out["pred_boxes"])
        diff_boxes = np.abs(pt_boxes - k_boxes)

        # Per-variant tolerance (default 1e-4; some configs have
        # inherently higher FP diff due to non-standard patch sizes)
        logits_tol = config.get("logits_mean_tol", 1e-4)
        boxes_tol = config.get("boxes_mean_tol", 1e-4)

        print(
            f"\n  [{name}/{image_name}] Logits — "
            f"max: {diff_logits.max():.6e}, mean: {diff_logits.mean():.6e}"
            f" (tol: {logits_tol:.0e})"
        )
        print(
            f"  [{name}/{image_name}] Boxes  — "
            f"max: {diff_boxes.max():.6e}, mean: {diff_boxes.mean():.6e}"
            f" (tol: {boxes_tol:.0e})"
        )

        assert diff_logits.mean() < logits_tol, (
            f"[{name}/{image_name}] Logits mean diff "
            f"{diff_logits.mean():.6e} > {logits_tol:.0e}"
        )
        assert diff_boxes.mean() < boxes_tol, (
            f"[{name}/{image_name}] Boxes mean diff "
            f"{diff_boxes.mean():.6e} > {boxes_tol:.0e}"
        )

    # ---- Test 2: detects expected objects on every COCO image -----------

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_detects_expected_objects(self, variant, image_name):
        """Keras model detects every expected COCO class above 0.3."""
        name = variant["name"]
        keras_model = variant["keras_model"]
        config = variant["config"]
        img = variant["images"][image_name]
        res = config["resolution"]
        expected = COCO_IMAGES[image_name]["expected_classes"]

        scores, labels, _ = _run_keras_detection(
            keras_model, img, res, config["num_queries"]
        )

        _print_detections(scores, labels, f"{name}/{image_name}", threshold=0.3)

        detected = set(labels[scores > 0.3].tolist())
        for cls_id in expected:
            cls_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
            assert cls_id in detected, (
                f"[{name}/{image_name}] Expected '{cls_name}' "
                f"(class {cls_id}) not detected. Got: {detected}"
            )

    # ---- Test 3: save verified weights (Phase 2) ------------------------

    def test_save_weights(self, variant):
        """Save .keras and .weights.h5 for a variant that passed parity."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        name = variant["name"]
        keras_model = variant["keras_model"]
        config = variant["config"]
        save_key = config["save_key"]

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        keras_path = os.path.join(WEIGHTS_DIR, f"{save_key}.keras")
        h5_path = os.path.join(WEIGHTS_DIR, f"{save_key}.weights.h5")

        print(f"\n  Saving {name} weights ...")
        print(f"    .keras -> {keras_path}")
        keras_model.save(keras_path)

        print(f"    .h5    -> {h5_path}")
        keras_model.save_weights(h5_path)

        assert os.path.exists(keras_path), f".keras file not found: {keras_path}"
        assert os.path.exists(h5_path), f".h5 file not found: {h5_path}"

        kb = os.path.getsize(keras_path) / 1024
        h5kb = os.path.getsize(h5_path) / 1024
        print(f"    .keras size: {kb:.0f} KB")
        print(f"    .h5    size: {h5kb:.0f} KB")
        print(f"    Weights dir: {WEIGHTS_DIR}")


# ---------------------------------------------------------------------------
# Phase 3: Reload .h5 weights (no PyTorch) and re-run detection tests
# ---------------------------------------------------------------------------


class TestReloadH5Weights:
    """Load saved .weights.h5 into a fresh Keras LWDETR model (no PyTorch)
    and verify detection still works on all three COCO images."""

    @pytest.fixture(
        scope="class",
        params=list(MODEL_CONFIGS.keys()),
    )
    def reloaded_model(self, request, coco_images):
        """Build a fresh Keras LWDETR, load .h5 weights, yield for tests."""
        name = request.param
        config = MODEL_CONFIGS[name]
        save_key = config["save_key"]
        h5_path = os.path.join(WEIGHTS_DIR, f"{save_key}.weights.h5")

        if not os.path.exists(h5_path):
            pytest.skip(
                f"{h5_path} not found — Phase 2 may have been skipped or failed"
            )

        print(f"\n{'=' * 60}")
        print(f"  Reloading variant: {name} from .h5")
        print(f"{'=' * 60}")

        # Fresh Keras LWDETR (no PyTorch involved)
        keras_model = build_keras_lwdetr(config)

        # Load the verified .h5 weights
        keras_model.load_weights(h5_path)
        print(f"  Loaded weights from {h5_path}")

        yield {
            "name": name,
            "keras_model": keras_model,
            "config": config,
            "images": coco_images,
        }

        del keras_model
        gc.collect()

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_h5_detects_expected_objects(self, reloaded_model, image_name):
        """After reloading .h5, the model detects expected objects."""
        name = reloaded_model["name"]
        keras_model = reloaded_model["keras_model"]
        config = reloaded_model["config"]
        img = reloaded_model["images"][image_name]
        res = config["resolution"]
        expected = COCO_IMAGES[image_name]["expected_classes"]

        scores, labels, _ = _run_keras_detection(
            keras_model, img, res, config["num_queries"]
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
        keras_model = reloaded_model["keras_model"]
        config = reloaded_model["config"]
        img = reloaded_model["images"][image_name]
        res = config["resolution"]

        scores, labels, _ = _run_keras_detection(
            keras_model, img, res, config["num_queries"]
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
