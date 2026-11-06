import os
import re
import tempfile
import functools
from types import SimpleNamespace
from logging import getLogger

import numpy as np
import h5py
from keras import ops

from paz.models.detection.dino_v2_object_detection.config import ModelConfig
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    LWDETR,
    CriterionArgs,
    post_process,
    apply_lwdetr,
)
from paz.models.detection.dino_v2_object_detection.models.backbone import (
    build_backbone,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (  # fmt: skip
    Transformer,
)
from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import (  # fmt: skip
    SegmentationHead,
)
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import (  # fmt: skip
    hungarian_matcher,
)

logger = getLogger(__name__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
KERAS_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "lwdetr_keras_weights")

WEIGHT_DECAY_EXEMPT = ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")  # fmt: skip
# Backbone-encoder (DINOv2 ViT) variables carry no parent scope in var.path:
# token/patch embeddings, transformer blocks, and the final norm. The
# projector (stage_*) and decoder/heads (lwdetr/*) are not encoder vars.
ENCODER_PATH_PREFIXES = ("embeddings", "encoder_layer_", "layernorm")
ENCODER_LAYER_PATTERN = re.compile(r"encoder_layer_(\d+)_")

IMAGENET_MEANS = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STDS = np.array([0.229, 0.224, 0.225], dtype="float32")
BBOX_LOSS_COEF = 5.0
GIOU_LOSS_COEF = 2.0
FOCAL_ALPHA = 0.25
MAX_MISSING_WEIGHTS = 30
RESTORE_TOLERANCE = 1e-5
UNSUPPORTED_FORMAT = "Unsupported weight format '{}'. Use .keras or .weights.h5 (or port from .pth first)."  # fmt: skip


def resolve_weights_path(filename):
    path = None
    candidate = None
    if filename is not None:
        candidate = os.path.join(KERAS_WEIGHTS_DIR, filename)
    if filename is not None and os.path.isfile(filename):
        path = filename
    elif candidate is not None and os.path.isfile(candidate):
        path = candidate
    return path


# The shipped lwdetr_*.weights.h5 files were saved from the previous subclass
# LWDETR (heads under attribute groups class_embed/bbox_embed/enc_out_*, the
# transformer under layers/functional, refpoint/query as root vars). The
# functional builder owns the same weights under different keys, so each
# unchanged sub-model subtree loads natively and the heads remap by name.
def assign_dense_from_group(dense, weights_file, group):
    dense.kernel.assign(weights_file[f"{group}/vars/0"][...])
    dense.bias.assign(weights_file[f"{group}/vars/1"][...])


def load_submodel_subtree(submodel, weights_file, group_name):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "subtree.weights.h5")
    with h5py.File(temp_path, "w") as temp_file:
        for child in weights_file[group_name].keys():
            source = weights_file[f"{group_name}/{child}"]
            weights_file.copy(source, temp_file, child)
    submodel.load_weights(temp_path)
    os.remove(temp_path)
    os.rmdir(temp_dir)


def legacy_dense_key(index):
    return "dense" if index == 0 else f"dense_{index}"


def load_bbox_head(model, weights_file):
    for index in range(3):
        layer = model.get_layer(f"bbox_embed_dense_{index}")
        group = f"bbox_embed/layers_list/{legacy_dense_key(index)}"
        assign_dense_from_group(layer, weights_file, group)


def load_encoder_heads(model, weights_file, group):
    layer = model.get_layer(f"enc_out_class_embed_{group}")
    class_group = f"enc_out_class_embed/{legacy_dense_key(group)}"
    assign_dense_from_group(layer, weights_file, class_group)
    mlp_key = "mlp" if group == 0 else f"mlp_{group}"
    for index in range(3):
        layer = model.get_layer(f"enc_out_bbox_embed_{group}_dense_{index}")
        key = legacy_dense_key(index)
        bbox_group = f"enc_out_bbox_embed/{mlp_key}/layers_list/{key}"
        assign_dense_from_group(layer, weights_file, bbox_group)


def load_detection_heads(model, weights_file):
    class_embed = model.get_layer("class_embed")
    assign_dense_from_group(class_embed, weights_file, "class_embed")
    load_bbox_head(model, weights_file)
    for group in range(model.group_detr):
        load_encoder_heads(model, weights_file, group)
    refpoints = model.get_layer("refpoint_embed").embeddings
    refpoints.assign(weights_file["vars/0"][...])
    queries = model.get_layer("query_feat").embeddings
    queries.assign(weights_file["vars/1"][...])


def is_legacy_checkpoint(weights_file):
    heads = "bbox_embed" in weights_file
    return heads and "layers_list" in weights_file["bbox_embed"]


def load_lwdetr_checkpoint(model, h5_path):
    with h5py.File(h5_path, "r") as weights_file:
        if not is_legacy_checkpoint(weights_file):
            model.load_weights(h5_path)
        else:
            load_submodel_subtree(model.backbone, weights_file, "backbone")
            group = "layers/functional"
            load_submodel_subtree(model.transformer, weights_file, group)
            load_detection_heads(model, weights_file)


def build_backbone_from_config(config):
    keys = ("encoder", "hidden_dim", "out_channels", "out_feature_indexes", "projector_scale", "layer_norm", "target_shape", "load_dinov2_weights", "patch_size", "num_windows", "positional_encoding_size")  # fmt: skip
    values = (config.encoder, config.hidden_dim, config.hidden_dim, config.out_feature_indexes, config.projector_scale, config.layer_norm, (config.resolution, config.resolution), config.pretrain_weights is None, config.patch_size, config.num_windows, config.positional_encoding_size)  # fmt: skip
    return build_backbone(**dict(zip(keys, values)))


def build_transformer_from_config(config):
    keys = ("d_model", "sa_nhead", "ca_nhead", "num_queries", "num_decoder_layers", "dim_feedforward", "dropout", "activation", "normalize_before", "return_intermediate_dec", "group_detr", "two_stage", "num_feature_levels", "dec_n_points", "lite_refpoint_refine", "decoder_norm_type", "bbox_reparam")  # fmt: skip
    values = (config.hidden_dim, config.sa_nheads, config.ca_nheads, config.num_queries, config.dec_layers, getattr(config, "dim_feedforward", 2048), 0.0, "relu", False, True, config.group_detr, config.two_stage, len(config.projector_scale), config.dec_n_points, config.lite_refpoint_refine, "LN", config.bbox_reparam)  # fmt: skip
    return Transformer(**dict(zip(keys, values)))


def build_segmentation_head_from_config(config):
    head = None
    if config.segmentation_head:
        keys = ("in_dim", "num_blocks", "downsample_ratio")
        values = (config.hidden_dim, config.dec_layers, config.mask_downsample_ratio)  # fmt: skip
        head = SegmentationHead(**dict(zip(keys, values)))
    return head


def build_matcher_from_config(config):
    keys = ("cost_class", "cost_bbox", "cost_giou", "focal_alpha")
    values = (getattr(config, "set_cost_class", 2), getattr(config, "set_cost_bbox", 5), getattr(config, "set_cost_giou", 2), getattr(config, "focal_alpha", 0.25))  # fmt: skip
    return functools.partial(hungarian_matcher, **dict(zip(keys, values)))


def build_model_from_config(config):
    keys = ("backbone", "transformer", "segmentation_head", "num_classes", "num_queries", "aux_loss", "group_detr", "two_stage", "lite_refpoint_refine", "bbox_reparam")  # fmt: skip
    values = (build_backbone_from_config(config), build_transformer_from_config(config), build_segmentation_head_from_config(config), config.num_classes + 1, config.num_queries, True, config.group_detr, config.two_stage, config.lite_refpoint_refine, config.bbox_reparam)  # fmt: skip
    return LWDETR(**dict(zip(keys, values)))


def build_base_weight_dict(config, train_config):
    keys = ("loss_ce", "loss_bbox", "loss_giou")
    values = (config.cls_loss_coef, BBOX_LOSS_COEF, GIOU_LOSS_COEF)
    weight_dict = dict(zip(keys, values))
    if config.segmentation_head and train_config is not None:
        mask_ce = getattr(train_config, "mask_ce_loss_coef", 5.0)
        weight_dict["loss_mask_ce"] = mask_ce
        weight_dict["loss_mask_dice"] = getattr(train_config, "mask_dice_loss_coef", 5.0)  # fmt: skip
    return weight_dict


def expand_weight_dict(weight_dict, config):
    # Iterate over the base keys only (not the growing dict) so auxiliary
    # entries do not cascade into duplicates.
    base = list(weight_dict.items())
    for index in range(config.dec_layers - 1):
        weight_dict.update({k + f"_{index}": v for k, v in base})
    if config.two_stage:
        weight_dict.update({k + "_enc": v for k, v in base})
    return weight_dict


def build_criterion_from_config(config, train_config=None):
    weight_dict = build_base_weight_dict(config, train_config)
    weight_dict = expand_weight_dict(weight_dict, config)
    losses = ["labels", "boxes", "cardinality"]
    if config.segmentation_head:
        losses.append("masks")
    keys = ("num_classes", "matcher", "weight_dict", "focal_alpha", "loss_types", "group_detr", "ia_bce_loss")  # fmt: skip
    values = (config.num_classes + 1, build_matcher_from_config(config), weight_dict, FOCAL_ALPHA, losses, config.group_detr, config.ia_bce_loss)  # fmt: skip
    criterion = CriterionArgs(**dict(zip(keys, values)))
    postprocess = functools.partial(post_process, num_select=config.num_select)
    return criterion, postprocess


def get_backbone_no_weight_decay_vars(model):
    # Only backbone-encoder (ViT) variables are eligible for exemption; the
    # projector (stage_*) and decoder/heads (lwdetr/*) are excluded by prefix.
    excluded = []
    for variable in model.trainable_variables:
        encoder = variable.path.startswith(ENCODER_PATH_PREFIXES)
        exempt = any(word in variable.path for word in WEIGHT_DECAY_EXEMPT)
        if encoder and exempt:
            excluded.append(variable)
    return excluded


def read_encoder_layer_id(path, num_layers):
    match = ENCODER_LAYER_PATTERN.search(path)
    if path.startswith("embeddings"):
        layer_id = 0
    elif match:
        layer_id = int(match.group(1)) + 1
    else:
        # Final layernorm or other non-block encoder params get decay 1.0.
        layer_id = num_layers + 1
    return layer_id


def compute_encoder_multiplier(path, rates, num_layers):
    lr, lr_encoder, lr_vit_layer_decay, lr_component_decay = rates
    layer_id = read_encoder_layer_id(path, num_layers)
    decay = lr_vit_layer_decay ** (num_layers + 1 - layer_id)
    return (lr_encoder / lr) * decay * (lr_component_decay**2)


def compute_variable_multiplier(path, rates, num_layers):
    if path.startswith(ENCODER_PATH_PREFIXES):
        multiplier = compute_encoder_multiplier(path, rates, num_layers)
    elif "transformer/decoder_" in path:
        multiplier = rates[3]
    else:
        multiplier = 1.0
    return multiplier


def get_param_lr_multipliers(model, train_config, model_config=None):
    rates = (train_config.lr, train_config.lr_encoder, train_config.lr_vit_layer_decay, train_config.lr_component_decay)  # fmt: skip
    indexes = model_config if model_config is not None else train_config
    num_layers = indexes.out_feature_indexes[-1] + 2
    multipliers = {}
    for variable in model.trainable_variables:
        args = (variable.path, rates, num_layers)
        multipliers[variable.path] = compute_variable_multiplier(*args)
    return multipliers


def load_weights_by_extension(model, weights_path):
    extension = os.path.splitext(weights_path)[-1].lower()
    if extension in (".h5", ".hdf5"):
        # Remaps the legacy subclass checkpoint onto the functional builder
        # and loads functional checkpoints straight through.
        load_lwdetr_checkpoint(model, weights_path)
    elif extension == ".keras":
        model.load_weights(weights_path)
    else:
        raise ValueError(UNSUPPORTED_FORMAT.format(extension))


def load_pretrained_weights(ns, weights_path=None):
    if weights_path is None:
        weights_path = ns.config.pretrain_weights
    if weights_path is not None:
        load_weights_by_extension(ns.model, weights_path)


def normalize_images(images):
    if images.ndim == 3:
        images = images[np.newaxis]
    return (images - IMAGENET_MEANS) / IMAGENET_STDS


def resize_to_resolution(images, resolution):
    tensor = ops.convert_to_tensor(images, dtype="float32")
    size = (resolution, resolution)
    return ops.image.resize(tensor, size, antialias=True)


def split_post_result(post_result):
    masks_list = post_result[3] if len(post_result) == 4 else None
    return post_result[0], post_result[1], post_result[2], masks_list


def format_prediction_results(scores, labels, boxes, masks_list, num, threshold):  # fmt: skip
    scores = ops.convert_to_numpy(scores)
    labels = ops.convert_to_numpy(labels)
    boxes = ops.convert_to_numpy(boxes)
    results = []
    for index in range(num):
        keep = scores[index] > threshold
        result = {"boxes": boxes[index][keep], "scores": scores[index][keep]}
        result["labels"] = labels[index][keep]
        if masks_list is not None:
            result["masks"] = ops.convert_to_numpy(masks_list[index])[keep]
        results.append(result)
    return results


def predict_detections(ns, images, threshold=0.5):
    images = normalize_images(images)
    resized = resize_to_resolution(images, ns.resolution)
    outputs = apply_lwdetr(ns.model, resized, training=False)
    sizes = np.array([[images.shape[1], images.shape[2]]] * images.shape[0])
    size_tensor = ops.convert_to_tensor(sizes, dtype="float32")
    scores, labels, boxes, masks = split_post_result(ns.postprocess(outputs, size_tensor))  # fmt: skip
    args = (scores, labels, boxes, masks, images.shape[0], threshold)
    return format_prediction_results(*args)


def snapshot_class_weights(model):
    weights = {}
    for weight in model.weights:
        if "class_embed" in weight.path:
            weights[weight.path] = weight.numpy().copy()
    return weights


def save_weights_to_temp(model):
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "reinit_checkpoint.weights.h5")
    model.save_weights(path)
    return directory, path


def rebuild_model_for_classes(ns, num_classes):
    ns.config = ns.config._replace(num_classes=num_classes)
    ns.model = build_model_from_config(ns.config)
    num_select = ns.config.num_select
    ns.postprocess = functools.partial(post_process, num_select=num_select)


def remove_temp_weights(directory, path):
    try:
        os.remove(path)
        os.rmdir(directory)
    except OSError:
        pass


def tile_to_shape(values, shape):
    if values.ndim == 2:
        repeats = int(np.ceil(shape[1] / values.shape[1]))
        tiled = np.tile(values, (1, repeats))[:, : shape[1]]
    elif values.ndim == 1:
        repeats = int(np.ceil(shape[0] / values.shape[0]))
        tiled = np.tile(values, repeats)[: shape[0]]
    else:
        tiled = None
    return tiled


def tile_class_weights(model, old_class_weights):
    for weight in model.weights:
        values = old_class_weights.get(weight.path)
        shape = tuple(weight.shape)
        tiled = None
        if values is not None and values.shape != shape:
            tiled = tile_to_shape(values, shape)
        if tiled is not None:
            weight.assign(tiled)
            args = (weight.path, values.shape, shape)
            logger.debug("Tiled class_embed weight %s: %s -> %s", *args)


def count_restored_weights(old_weights, old_shapes, new_weights, new_shapes):
    restored, shape_changed = 0, 0
    for index in range(min(len(old_weights), len(new_weights))):
        difference = np.inf
        if old_shapes[index] == new_shapes[index]:
            difference = np.max(np.abs(old_weights[index] - new_weights[index]))  # fmt: skip
        if old_shapes[index] != new_shapes[index]:
            shape_changed += 1
        elif float(difference) < RESTORE_TOLERANCE:
            restored += 1
    return restored, shape_changed


def guard_restored_weights(old_count, restored, shape_changed):
    minimum = old_count - MAX_MISSING_WEIGHTS
    if restored < minimum:
        message = f"Too few weights restored: {restored} < {minimum}. "
        message += f"shape_changed={shape_changed}. "
        raise RuntimeError(message + "Possible architecture mismatch between builds.")  # fmt: skip


def warn_missing_reinitialization(num_classes, old_num_classes, shape_changed):
    if num_classes != old_num_classes and shape_changed == 0:
        message = "reinitialize_detection_head: num_classes changed but no "
        logger.warning(message + "weight shapes differed - head may not have been reinitialised.")  # fmt: skip


def reinitialize_detection_head(ns, num_classes):
    old_num_classes = ns.config.num_classes
    old_weights = [w.numpy().copy() for w in ns.model.weights]
    old_shapes = [tuple(w.shape) for w in ns.model.weights]
    old_class_weights = snapshot_class_weights(ns.model)
    directory, path = save_weights_to_temp(ns.model)
    rebuild_model_for_classes(ns, num_classes)
    ns.model.load_weights(path, skip_mismatch=True)
    remove_temp_weights(directory, path)
    tile_class_weights(ns.model, old_class_weights)
    new_weights = [w.numpy() for w in ns.model.weights]
    new_shapes = [tuple(w.shape) for w in ns.model.weights]
    warn_weight_count_change(len(old_weights), len(new_weights))
    args = (old_weights, old_shapes, new_weights, new_shapes)
    restored, shape_changed = count_restored_weights(*args)
    guard_restored_weights(len(old_weights), restored, shape_changed)
    warn_missing_reinitialization(num_classes, old_num_classes, shape_changed)
    report_reinitialization(restored, shape_changed, len(new_weights))


def warn_weight_count_change(old_count, new_count):
    if new_count != old_count:
        message = "reinitialize_detection_head: weight count changed from %d to %d"  # fmt: skip
        logger.warning(message, old_count, new_count)


def report_reinitialization(restored, shape_changed, total):
    message = "reinitialize_detection_head: restored=%d, shape_changed=%d (of %d total)"  # fmt: skip
    logger.info(message, restored, shape_changed, total)


def auto_load_weights(ns, config):
    path = resolve_weights_path(config.pretrain_weights)
    if path is not None:
        ns.load_pretrained_weights(path)
    else:
        message = "Pretrained weights '%s' not found. Model initialised with random weights."  # fmt: skip
        logger.warning(message, config.pretrain_weights)


def Model(config):
    if not isinstance(config, ModelConfig):
        raise TypeError(f"Expected ModelConfig, got {type(config)}")
    ns = SimpleNamespace()
    ns.config = config
    ns.resolution = config.resolution
    ns.model = build_model_from_config(config)
    ns.postprocess = functools.partial(post_process, num_select=config.num_select)  # fmt: skip
    ns.class_names = None
    keys = ("load_pretrained_weights", "predict", "reinitialize_detection_head")  # fmt: skip
    functions = (load_pretrained_weights, predict_detections, reinitialize_detection_head)  # fmt: skip
    for key, function in zip(keys, functions):
        setattr(ns, key, functools.partial(function, ns))
    # The functional builder already materialised every weight, so a ported
    # .weights.h5 / .keras checkpoint loads directly by name.
    if config.pretrain_weights is not None:
        auto_load_weights(ns, config)
    return ns
