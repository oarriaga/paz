import gc
import io
import math
import os
import sys

import numpy as np
import pytest
from urllib.request import urlopen

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reference implementation guard
try:
    import torch
    import torchvision.transforms.functional as F_tv
    from PIL import Image

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Reference RFDETR imports (detection only)
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
                current_dir,
                "../../../../../../examples/"
                "rf-detr_original_pytorch_implementation",
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

# Keras imports
from keras import ops
import functools

# LWDETR model imports
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    LWDETR,
    post_process,
    apply_lwdetr,
)
from paz.models.detection.dino_v2_object_detection.main import (
    load_lwdetr_checkpoint,
)
from paz.models.detection.dino_v2_object_detection.models.backbone import (
    build_backbone as build_keras_backbone,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (  # fmt: skip
    Transformer as KerasTransformer,
)

# Weight transfer utilities
from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (  # fmt: skip
    transfer_encoder as transfer_backbone_encoder,
    port_weights_multiscale_projector,
    transfer_layernorm,
    optional_embedding_table,
    assign_table,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer_weights_porting_utils import (  # fmt: skip
    transfer_transformer_weights,
)

# COCO class labels
from paz.models.detection.dino_v2_object_detection.utils.coco_classes import (
    COCO_CLASSES,
)

# Constants

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

# Model configurations — detection only (no segmentation)

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
AVAILABLE_VARIANTS = [name for name, config in MODEL_CONFIGS.items() if config.get("pt_class") is not None]  # fmt: skip


# Weight transfer helpers


def read_torch_pos_embed(pt_embeddings_layer):
    pos_embed = pt_embeddings_layer.position_embeddings
    source = pos_embed.weight if hasattr(pos_embed, "weight") else pos_embed
    values = source.detach().cpu().numpy()
    if values.ndim == 2:
        values = np.expand_dims(values, axis=0)
    return values


def interpolate_grid_tokens(grid_tokens, grid_size, target_size):
    reshaped = grid_tokens.reshape(1, grid_size, grid_size, -1)
    # Bicubic interpolation matches the DINOv2 runtime code in
    # dinov2_with_windowed_attn.py::interpolate_pos_encoding
    tensor = torch.tensor(reshaped).permute(0, 3, 1, 2)
    tensor = tensor.to(dtype=torch.float32)
    keys = ("size", "mode", "align_corners", "antialias")
    values = ((target_size, target_size), "bicubic", False, True)
    kwargs = dict(zip(keys, values))
    resized = torch.nn.functional.interpolate(tensor, **kwargs)
    return resized.permute(0, 2, 3, 1).numpy()


def resize_pos_embed(pt_pos_embed, keras_shape):
    grid_tokens = pt_pos_embed[:, 1:, :]
    num_tokens = grid_tokens.shape[1]
    resized = None
    if num_tokens == 0:
        print("  WARNING: PyTorch grid tokens are empty - skipping resize.")
    else:
        args = (grid_tokens, int(np.sqrt(num_tokens)))
        target = int(np.sqrt(keras_shape[0] - 1))
        interpolated = interpolate_grid_tokens(*args, target)
        last_dim = pt_pos_embed.shape[-1]
        interpolated = interpolated.reshape(1, -1, last_dim)
        parts = [pt_pos_embed[:, 0:1, :], interpolated]
        resized = np.concatenate(parts, axis=1)
    return resized


def resize_and_assign_pos_embed(pt_embeddings_layer, keras_pos):
    pt_pos_embed = read_torch_pos_embed(pt_embeddings_layer)
    keras_shape = keras_pos.shape
    if pt_pos_embed.shape[1] == keras_shape[0]:
        keras_pos.assign(np.reshape(pt_pos_embed, keras_shape))
        return
    shapes = f"PT {pt_pos_embed.shape} -> Keras {keras_shape}"
    print(f"  Resizing PosEmbed: {shapes}")
    resized = resize_pos_embed(pt_pos_embed, keras_shape)
    if resized is not None:
        keras_pos.assign(np.reshape(resized, keras_shape))


def set_dense_from_torch(keras_dense, torch_linear):
    keras_dense.set_weights(
        [
            torch_linear.weight.detach().cpu().numpy().T,
            torch_linear.bias.detach().cpu().numpy(),
        ]
    )


def torch_array(param):
    source = param.weight if hasattr(param, "weight") else param
    return source.detach().cpu().numpy()


def transfer_encoder_output_heads(pt_model, keras_model, group_detr):
    for group in range(group_detr):
        layer = keras_model.get_layer(f"enc_out_class_embed_{group}")
        torch_head = pt_model.transformer.enc_out_class_embed[group]
        set_dense_from_torch(layer, torch_head)
        torch_bbox = pt_model.transformer.enc_out_bbox_embed[group]
        for index, torch_layer in enumerate(torch_bbox.layers):
            name = f"enc_out_bbox_embed_{group}_dense_{index}"
            set_dense_from_torch(keras_model.get_layer(name), torch_layer)


def transfer_query_embeddings(pt_model, keras_model):
    refpoints = keras_model.get_layer("refpoint_embed").embeddings
    refpoints.assign(torch_array(pt_model.refpoint_embed))
    queries = keras_model.get_layer("query_feat").embeddings
    queries.assign(torch_array(pt_model.query_feat))


def transfer_lwdetr_head_weights(pt_model, keras_model, config):
    class_embed = keras_model.get_layer("class_embed")
    set_dense_from_torch(class_embed, pt_model.class_embed)
    for index, torch_layer in enumerate(pt_model.bbox_embed.layers):
        layer = keras_model.get_layer(f"bbox_embed_dense_{index}")
        set_dense_from_torch(layer, torch_layer)
    transfer_query_embeddings(pt_model, keras_model)
    if config.get("two_stage", True):
        group_detr = config.get("group_detr", 13)
        transfer_encoder_output_heads(pt_model, keras_model, group_detr)


def precompute_pos_embed_interpolation(pt_backbone, config):
    embeddings = pt_backbone.encoder.encoder.embeddings
    stored_grid = int(math.sqrt(embeddings.position_embeddings.shape[1] - 1))
    target_grid = config["resolution"] // config["patch_size"]
    # export() bakes the interpolation in when the pretrained grid differs
    # from the target grid (e.g. 37x37 vs 40x40); the Keras model is already
    # built at the target size, so afterwards this is a direct copy.
    if stored_grid != target_grid:
        sizes = f"{stored_grid}x{stored_grid} -> {target_grid}x{target_grid}"
        print(f"  Pre-computing pos embed interpolation: {sizes}")
        pt_backbone.encoder.export()


def read_patch_projection(pt_patch_embed):
    if hasattr(pt_patch_embed, "projection"):
        projection = pt_patch_embed.projection
    elif hasattr(pt_patch_embed, "proj"):
        projection = pt_patch_embed.proj
    else:
        raise AttributeError(f"Could not find projection weights in {pt_patch_embed}")  # fmt: skip
    return projection


def transfer_patch_embeddings(pt_embeddings, k_model):
    patch_embed = pt_embeddings
    if hasattr(pt_embeddings, "patch_embeddings"):
        patch_embed = pt_embeddings.patch_embeddings
    projection = read_patch_projection(patch_embed)
    keras_projection = k_model.get_layer("embeddings_patch_embeddings_projection")  # fmt: skip
    kernel = projection.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
    keras_projection.kernel.assign(kernel)
    keras_projection.bias.assign(projection.bias.detach().cpu().numpy())


def transfer_special_tokens(pt_embeddings, k_model):
    if hasattr(pt_embeddings, "cls_token"):
        cls_table = k_model.get_layer("embeddings_cls_token").embeddings
        assign_table(cls_table, pt_embeddings.cls_token.detach().cpu().numpy())
    mask_token = optional_embedding_table(k_model, "embeddings_mask_token")
    if mask_token is not None and hasattr(pt_embeddings, "mask_token"):
        assign_table(mask_token, pt_embeddings.mask_token.detach().cpu().numpy())  # fmt: skip


def transfer_backbone_weights(pt_backbone, keras_backbone, config):
    k_model = keras_backbone.get_layer("encoder")
    precompute_pos_embed_interpolation(pt_backbone, config)
    embeddings = pt_backbone.encoder.encoder.embeddings
    layer = k_model.get_layer("embeddings_position_embeddings")
    resize_and_assign_pos_embed(embeddings, layer.embeddings)
    transfer_patch_embeddings(embeddings, k_model)
    transfer_special_tokens(embeddings, k_model)
    encoder = pt_backbone.encoder.encoder
    transfer_backbone_encoder(encoder.encoder, k_model, "encoder")
    transfer_layernorm(encoder.layernorm, k_model.get_layer("layernorm"))
    projector = keras_backbone.get_layer("projector")
    port_weights_multiscale_projector(pt_backbone.projector, projector)


def transfer_full_model_weights(pt_model, keras_model, config):
    inner_pt = pt_model.model.model
    keras_backbone = keras_model.backbone.get_layer("backbone")
    args = (inner_pt.backbone[0], keras_backbone, config)
    transfer_backbone_weights(*args)
    args = (inner_pt.transformer, keras_model.transformer)
    transfer_transformer_weights(*args, config["hidden_dim"], config["sa_nheads"])  # fmt: skip
    transfer_lwdetr_head_weights(inner_pt, keras_model, config)
    print("  Weight transfer complete.")


# Keras model builder


def build_porting_backbone(config):
    keys = ("encoder", "hidden_dim", "out_channels", "patch_size", "num_windows", "out_feature_indexes", "projector_scale", "layer_norm", "target_shape", "positional_encoding_size")  # fmt: skip
    resolution = config["resolution"]
    values = (config["encoder"], config["hidden_dim"], config["hidden_dim"], config["patch_size"], config["num_windows"], config["out_feature_indexes"], config["projector_scale"], True, (resolution, resolution), config.get("positional_encoding_size", 37))  # fmt: skip
    return build_keras_backbone(**dict(zip(keys, values)))


def build_porting_transformer(config):
    keys = ("d_model", "sa_nhead", "ca_nhead", "num_queries", "num_decoder_layers", "num_feature_levels", "dec_n_points", "two_stage", "bbox_reparam", "return_intermediate_dec", "lite_refpoint_refine", "group_detr")  # fmt: skip
    values = (config["hidden_dim"], config["sa_nheads"], config["ca_nheads"], config["num_queries"], config["dec_layers"], len(config["projector_scale"]), config["dec_n_points"], True, True, True, config.get("lite_refpoint_refine", True), config.get("group_detr", 13))  # fmt: skip
    return KerasTransformer(**dict(zip(keys, values)))


def build_keras_lwdetr(config):
    keys = ("backbone", "transformer", "segmentation_head", "num_classes", "num_queries", "group_detr", "two_stage", "bbox_reparam", "lite_refpoint_refine")  # fmt: skip
    values = (build_porting_backbone(config), build_porting_transformer(config), None, config.get("num_classes", 91), config["num_queries"], config.get("group_detr", 13), True, True, config.get("lite_refpoint_refine", True))  # fmt: skip
    model = LWDETR(**dict(zip(keys, values)))
    # Exercise the functional model once: every group_detr head is
    # materialised at build time and training=True runs all of them.
    resolution = config["resolution"]
    dummy = np.ones((1, resolution, resolution, 3), dtype=np.float32) * 0.5
    apply_lwdetr(model, dummy, training=True)
    return model


# Helpers


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def download_coco_image(image_id, url):
    ensure_cache_dir()
    cached = os.path.join(CACHE_DIR, f"coco_val_{image_id}.npy")
    if os.path.exists(cached):
        image = np.load(cached)
    else:
        print(f"  Downloading COCO image {image_id} ...")
        data = urlopen(url).read()
        decoded = Image.open(io.BytesIO(data)).convert("RGB")
        image = np.array(decoded, dtype=np.uint8)
        np.save(cached, image)
    return image


def run_reference_forward(pt_model, preprocessed, resolution):
    pt_input = torch.from_numpy(preprocessed).permute(0, 3, 1, 2)
    mask = torch.zeros((1, resolution, resolution), dtype=torch.bool)
    with torch.no_grad():
        outputs = pt_model.model.model(NestedTensor(pt_input, mask))
    return outputs


def compare_parity_field(pt_out, k_out, key, tag, label, tolerance):
    reference = pt_out[key].cpu().numpy()
    difference = np.abs(reference - ops.convert_to_numpy(k_out[key]))
    summary = f"max: {difference.max():.6e}, mean: {difference.mean():.6e}"
    print(f"\n  [{tag}] {label} - {summary} (tol: {tolerance:.0e})")
    message = f"[{tag}] {label} mean diff {difference.mean():.6e} > {tolerance:.0e}"  # fmt: skip
    assert difference.mean() < tolerance, message


def preprocess_image(image_float, resolution):
    # antialias=False keeps the resize matching tf.image.resize semantics
    t = F_tv.to_tensor(image_float)  # (3,H,W)
    means = IMAGENET_MEANS.tolist()
    stds = IMAGENET_STDS.tolist()
    t = F_tv.normalize(t, means, stds)  # normalise
    t = F_tv.resize(t, [resolution, resolution], antialias=False)  # resize
    return t.unsqueeze(0).permute(0, 2, 3, 1).numpy()  # (1,H,W,3)


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


def run_keras_detection(keras_lwdetr, image_float, resolution, num_select=300):
    preprocessed = preprocess_image(image_float, resolution)
    raw = apply_lwdetr(keras_lwdetr, preprocessed, training=False)
    H, W = image_float.shape[:2]
    pp = functools.partial(post_process, num_select=num_select)
    scores, labels, boxes = pp(
        raw,
        ops.convert_to_tensor(np.array([[H, W]], dtype="float32")),
    )
    return (
        ops.convert_to_numpy(scores)[0],
        ops.convert_to_numpy(labels)[0],
        ops.convert_to_numpy(boxes)[0],
    )


# Fixtures


@pytest.fixture(scope="session")
def coco_images():
    images = {}
    for name, info in COCO_IMAGES.items():
        arr = download_coco_image(info["id"], info["url"])
        images[name] = arr.astype("float32") / 255.0
    return images


# Phase 1: Build Keras LWDETR, port weights, verify output parity


def force_eager_attention(pt_model):
    # Eager attention matches the Keras matmul -> softmax -> matmul path;
    # SDPA kernels diverge in FP and break the sub-1e-4 parity check.
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


def build_and_port_variant(variant_name):
    config = MODEL_CONFIGS[variant_name]

    # 1. Instantiate reference model (auto-downloads weights)
    print(f"\n  Instantiating reference {variant_name}...")
    if "XLarge" in variant_name or "Xlarge" in variant_name:
        pt_model = config["pt_class"](accept_platform_model_license=True)
    else:
        pt_model = config["pt_class"]()
    pt_model.model.model.eval()
    pt_model.model.model.cpu()

    # Force eager attention to match the Keras matmul -> softmax ->
    # matmul sequence, eliminating attention-kernel FP divergence.
    force_eager_attention(pt_model)

    # 2. Build Keras LWDETR
    print(f"  Building Keras LWDETR for {variant_name}...")
    keras_model = build_keras_lwdetr(config)

    # 3. Transfer weights
    print(f"  Transferring weights for {variant_name}...")
    transfer_full_model_weights(pt_model, keras_model, config)

    return pt_model, keras_model, config


_NO_TORCH_REASON = "Reference implementation not installed"


@pytest.mark.skipif(not HAS_TORCH, reason=_NO_TORCH_REASON)
class TestPortingParity:

    @pytest.fixture(
        scope="class",
        params=[v for v in AVAILABLE_VARIANTS],
    )
    def variant(self, request, coco_images):
        name = request.param
        print(f"\n{'=' * 60}")
        print(f"  Building variant: {name}")
        print(f"{'=' * 60}")

        pt_model, keras_model, config = build_and_port_variant(name)

        yield {
            "name": name,
            "pt_model": pt_model,
            "keras_model": keras_model,
            "config": config,
            "images": coco_images,
        }

        # Teardown: free reference model
        del pt_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_forward_parity(self, variant, image_name):
        config = variant["config"]
        resolution = config["resolution"]
        image = variant["images"][image_name]
        preprocessed = preprocess_image(image, resolution)
        args = (variant["pt_model"], preprocessed, resolution)
        pt_out = run_reference_forward(*args)
        keras_model = variant["keras_model"]
        k_out = apply_lwdetr(keras_model, preprocessed, training=False)
        tag = f"{variant['name']}/{image_name}"
        # Some configs carry inherently higher FP diff because of their
        # non-standard patch sizes, so tolerances are per variant.
        tolerance = config.get("logits_mean_tol", 1e-4)
        args = (pt_out, k_out, "pred_logits", tag, "Logits")
        compare_parity_field(*args, tolerance)
        tolerance = config.get("boxes_mean_tol", 1e-4)
        args = (pt_out, k_out, "pred_boxes", tag, "Boxes")
        compare_parity_field(*args, tolerance)

    @pytest.mark.parametrize("image_name", list(COCO_IMAGES.keys()))
    def test_detects_expected_objects(self, variant, image_name):
        name = variant["name"]
        keras_model = variant["keras_model"]
        config = variant["config"]
        image = variant["images"][image_name]
        res = config["resolution"]
        expected = COCO_IMAGES[image_name]["expected_classes"]

        scores, labels, _ = run_keras_detection(
            keras_model, image, res, config["num_queries"]
        )

        print_detections(scores, labels, f"{name}/{image_name}", threshold=0.3)

        detected = set(labels[scores > 0.3].tolist())
        for cls_id in expected:
            cls_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
            assert cls_id in detected, (
                f"[{name}/{image_name}] Expected '{cls_name}' "
                f"(class {cls_id}) not detected. Got: {detected}"
            )

    def test_save_weights(self, variant):
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

        keras_msg = f".keras file not found: {keras_path}"
        assert os.path.exists(keras_path), keras_msg
        assert os.path.exists(h5_path), f".h5 file not found: {h5_path}"

        kb = os.path.getsize(keras_path) / 1024
        h5kb = os.path.getsize(h5_path) / 1024
        print(f"    .keras size: {kb:.0f} KB")
        print(f"    .h5    size: {h5kb:.0f} KB")
        print(f"    Weights dir: {WEIGHTS_DIR}")


# Phase 3: Reload .h5 weights and re-run detection tests


class TestReloadH5Weights:

    @pytest.fixture(
        scope="class",
        params=list(MODEL_CONFIGS.keys()),
    )
    def reloaded_model(self, request, coco_images):
        name = request.param
        config = MODEL_CONFIGS[name]
        save_key = config["save_key"]
        h5_path = os.path.join(WEIGHTS_DIR, f"{save_key}.weights.h5")

        if not os.path.exists(h5_path):
            pytest.skip(
                f"{h5_path} not found — Phase 2 may have "
                "been skipped or failed"
            )

        print(f"\n{'=' * 60}")
        print(f"  Reloading variant: {name} from .h5")
        print(f"{'=' * 60}")

        keras_model = build_keras_lwdetr(config)

        # Load verified .h5 weights (legacy or functional format)
        load_lwdetr_checkpoint(keras_model, h5_path)
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
        name = reloaded_model["name"]
        keras_model = reloaded_model["keras_model"]
        config = reloaded_model["config"]
        image = reloaded_model["images"][image_name]
        res = config["resolution"]
        expected = COCO_IMAGES[image_name]["expected_classes"]

        scores, labels, _ = run_keras_detection(
            keras_model, image, res, config["num_queries"]
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
    def test_h5_has_confident_detections(self, reloaded_model, image_name):
        name = reloaded_model["name"]
        keras_model = reloaded_model["keras_model"]
        config = reloaded_model["config"]
        image = reloaded_model["images"][image_name]
        res = config["resolution"]

        scores, labels, _ = run_keras_detection(
            keras_model, image, res, config["num_queries"]
        )

        n = int((scores > 0.3).sum())
        assert n > 0, (
            f"[h5-reload/{name}/{image_name}] No detections > 0.3 "
            f"after .h5 reload"
        )


# Entry point

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
