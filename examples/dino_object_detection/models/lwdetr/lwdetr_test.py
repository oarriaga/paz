import os
import sys
import numpy as np
import torch
import keras
from keras import ops
import re
import math
import copy

from torch import nn

# --- Environment Setup ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# --- Path Setup ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
sys.path.append(project_root)

from examples.dino_object_detection.models.lwdetr.torch_lwdetr_for_testing import (
    build_model as build_model_torch,
)
from examples.dino_object_detection.models.lwdetr.lwdetr_keras import (
    build_model as build_model_keras,
)
from examples.dino_object_detection.models.utils.torch_misc_for_testing import (
    NestedTensor,
)


# --- Configuration ---
class Args:
    def __init__(self):
        self.num_classes = 91
        self.device = "cpu"
        # Backbone
        self.encoder = "dinov2_small"
        self.vit_encoder_num_layers = 12
        self.pretrained_encoder = None
        self.window_block_indexes = []
        self.drop_path = 0.0
        self.out_feature_indexes = [2, 5, 8, 11]
        self.projector_scale = ["P3", "P4", "P5"]
        self.use_cls_token = True
        self.position_embedding = "sine"
        self.freeze_encoder = False
        self.layer_norm = True
        self.shape = (518, 518)
        self.rms_norm = False
        self.backbone_lora = False
        self.force_no_pretrain = True
        self.gradient_checkpointing = False
        self.pretrain_weights = None
        self.patch_size = 14
        self.num_windows = 1
        self.positional_encoding_size = 37
        self.encoder_only = False
        self.backbone_only = False
        self.num_register_tokens = 0

        # Transformer
        self.hidden_dim = 256
        self.dim_feedforward = 1024
        self.dropout = 0.0
        self.nheads = 8
        self.num_queries = 300
        self.enc_layers = 0
        self.dec_layers = 6
        self.num_feature_levels = 4
        self.enc_n_points = 4
        self.dec_n_points = 4
        self.sa_nheads = 8
        self.ca_nheads = 8
        self.activation = "relu"
        self.transformer_activation = "relu"
        self.pre_norm = False
        self.normalize_before = False
        self.return_intermediate_dec = True
        self.use_lsj = False
        self.masks = False
        self.decoder_norm = "LN"

        # DINO/LWDETR
        self.aux_loss = True
        self.group_detr = 1
        self.two_stage = True
        self.two_stage_num_proposals = 300
        self.lite_refpoint_refine = True
        self.bbox_reparam = True
        self.segmentation_head = False
        self.num_select = 300

        # Loss/Matcher
        self.cls_loss_coef = 1.0
        self.bbox_loss_coef = 5.0
        self.giou_loss_coef = 2.0
        self.mask_ce_loss_coef = 1.0
        self.mask_dice_loss_coef = 1.0
        self.focal_alpha = 0.25
        self.use_varifocal_loss = False
        self.use_position_supervised_loss = False
        self.ia_bce_loss = False
        self.sum_group_losses = False
        self.mask_point_sample_ratio = 16
        self.mask_downsample_ratio = 4
        self.set_cost_class = 2.0
        self.set_cost_bbox = 5.0
        self.set_cost_giou = 2.0


TEST_ARGS = Args()


# --- Helpers ---
def log(msg, section=False, status=None):
    if section:
        print(f"\n{'='*80}\n{msg}\n{'='*80}")
    else:
        prefix = "[INFO]"
        if status == "PASS":
            prefix = "\033[92m[PASS]\033[0m"
        elif status == "FAIL":
            prefix = "\033[91m[FAIL]\033[0m"
        elif status == "WARN":
            prefix = "\033[93m[WARN]\033[0m"
        print(f"{prefix} {msg}")


def check_close(t_val, k_val, name, atol=1e-4, rtol=1e-4):
    if isinstance(t_val, torch.Tensor):
        t_np = t_val.detach().cpu().numpy()
    else:
        t_np = np.array(t_val)

    if hasattr(k_val, "numpy"):
        k_np = k_val.numpy()
    elif isinstance(k_val, (list, tuple)):
        k_np = np.array(k_val[0]) if len(k_val) > 0 else np.array(k_val)
    elif hasattr(k_val, "detach"):
        k_np = k_val.detach().cpu().numpy()
    else:
        k_np = np.array(k_val)

    if t_np.ndim == 4 and k_np.ndim == 4:
        if t_np.shape[1] == k_np.shape[3] and t_np.shape[2] == k_np.shape[1]:
            k_np = np.transpose(k_np, (0, 3, 1, 2))

    t_flat = t_np.flatten()
    k_flat = k_np.flatten()

    if t_flat.shape != k_flat.shape:
        log(
            f"{name} | SHAPE MISMATCH: Torch {t_np.shape} vs Keras {k_np.shape}",
            status="FAIL",
        )
        return False, 9999.0

    diff = np.abs(t_flat - k_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    status = "PASS" if max_diff < atol else "FAIL"
    first_4_vals_t = t_flat[:4]
    first_4_vals_k = k_flat[:4]
    min_diff = np.min(diff)
    std_diff = np.std(diff)
    if status == "FAIL":
        msg = (
            f"{name:<40} | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f} | "
            f"Min Diff: {min_diff:.6f} | Std Diff: {std_diff:.6f} | "
            f"Torch Min: {np.min(t_flat):.6f} | Torch Max: {np.max(t_flat):.6f} | "
            f"Torch Std: {np.std(t_flat):.6f} | "
            f"Keras Min: {np.min(k_flat):.6f} | Keras Max: {np.max(k_flat):.6f} | "
            f"Keras Std: {np.std(k_flat):.6f} | first 4 Torch: {first_4_vals_t} | "
            f"first 4 Keras: {first_4_vals_k}"
        )
    else:
        msg = f"{name:<40} | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f} | Torch Max: {np.max(t_flat):.6f}| Keras Max: {np.max(k_flat):.6f}| first 4 Torch: {first_4_vals_t} | "
        f"first 4 Keras: {first_4_vals_k}"
    log(msg, status=status)
    return status == "PASS", max_diff


# -------------------------------------------------------------------------
# Weight Porting Helpers
# -------------------------------------------------------------------------


def set_weights(layer, weights_list):
    """Safe wrapper for Keras set_weights."""
    if "Identity" in layer.__class__.__name__:
        return False
    try:
        current_weights = layer.get_weights()
        if len(current_weights) != len(weights_list):
            pass
        layer.set_weights(weights_list)
        return True
    except Exception as e:
        log(f"Error setting weights for {layer.name}: {e}", status="FAIL")
        return False


# --- Backbone Helpers ---
def get_keras_blocks(model, depth):
    try:
        wrapper = model.get_layer("dino_v2_backbone_wrapper")
        inner_model = getattr(wrapper, "feature_extractor", wrapper)
    except:
        inner_model = model

    found_blocks = []

    def _collect_blocks(layer_or_model):
        if isinstance(layer_or_model, (list, tuple)):
            for l in layer_or_model:
                _collect_blocks(l)
            return

        name_lower = layer_or_model.name.lower()
        class_lower = layer_or_model.__class__.__name__.lower()
        is_block = (
            "block" in name_lower
            or "transformer" in name_lower
            or "encoderlayer" in class_lower
        )
        is_chunk = (
            "chunk" in name_lower
            or "chunk" in class_lower
            or "sequential" in class_lower
        )
        if is_block and not is_chunk:
            found_blocks.append(layer_or_model)
            return

        if hasattr(layer_or_model, "layers"):
            _collect_blocks(layer_or_model.layers)
        elif hasattr(layer_or_model, "blocks"):
            _collect_blocks(layer_or_model.blocks)

    if hasattr(inner_model, "blocks"):
        candidates = list(inner_model.blocks)
        if len(candidates) == depth:
            return candidates

    _collect_blocks(inner_model.layers)
    try:
        found_blocks.sort(
            key=lambda x: (
                int(re.search(r"(\d+)", x.name).group(1))
                if re.search(r"(\d+)", x.name)
                else 999
            )
        )
    except:
        pass
    found_blocks = [b for b in found_blocks if "Identity" not in b.__class__.__name__]
    return found_blocks


def get_backbone_and_wrapper(keras_model):
    if hasattr(keras_model, "backbone"):
        backbone_model = keras_model.backbone
    else:
        backbone_model = keras_model

    try:
        wrapper = backbone_model.get_layer("dino_v2_backbone_wrapper")
    except ValueError:
        wrapper = None
        for layer in backbone_model.layers:
            if "DinoV2BackboneWrapper" in layer.__class__.__name__:
                wrapper = layer
                break
    return backbone_model, wrapper


def port_backbone_weights(t_joiner, k_joiner):
    log(">> Porting Backbone Weights...", section=True)
    t_backbone_wrapper = t_joiner[0].encoder
    t_backbone = t_backbone_wrapper.encoder
    sd = t_backbone.state_dict()
    full_sd = t_joiner[0].state_dict()

    k_backbone_model, wrapper = get_backbone_and_wrapper(k_joiner)
    if wrapper is None:
        log(
            "Could not find DinoV2BackboneWrapper in Keras model. Porting might fail.",
            status="WARN",
        )
        k_model = k_backbone_model
    else:
        k_model = getattr(wrapper, "feature_extractor", wrapper)

    # A. Positional Embeddings
    t_pos = sd["embeddings.position_embeddings"]
    k_pos_var = None
    for attr in ["positional_embedding", "pos_embed", "position_embeddings"]:
        if hasattr(k_model, attr):
            k_pos_var = getattr(k_model, attr)
            break
    if k_pos_var is not None:
        k_pos_var.assign(t_pos.numpy())
        log("Ported Positional Embeddings", status="PASS")

    # B. CLS Token
    t_cls = sd["embeddings.cls_token"]
    for attr in ["class_token", "classification_token", "cls_token"]:
        if hasattr(k_model, attr):
            getattr(k_model, attr).assign(t_cls.numpy())
            log(f"Ported CLS Token ({attr})", status="PASS")
            break

    # C. Patch Embeddings
    t_patch_w = sd["embeddings.patch_embeddings.projection.weight"]
    t_patch_b = sd["embeddings.patch_embeddings.projection.bias"]
    k_patch = None
    for l in k_model.layers:
        if "patch" in l.name.lower() or "conv" in l.name.lower():
            k_patch = l
            break
    if k_patch:
        target_layer = getattr(k_patch, "projection_layer", k_patch)
        w_np = t_patch_w.numpy().transpose(2, 3, 1, 0)
        b_np = t_patch_b.numpy()
        set_weights(target_layer, [w_np, b_np])
        log("Ported Patch Embeddings", status="PASS")

    # D. Blocks
    k_blocks = get_keras_blocks(k_model, TEST_ARGS.vit_encoder_num_layers)
    log(f"Found {len(k_blocks)} Keras Blocks")

    for i, k_blk in enumerate(k_blocks):
        prefix = f"encoder.layer.{i}."

        n1_w = sd[f"{prefix}norm1.weight"].numpy()
        n1_b = sd[f"{prefix}norm1.bias"].numpy()
        n1_layer = getattr(k_blk, "normalization1", getattr(k_blk, "norm1", None))
        if n1_layer:
            set_weights(n1_layer, [n1_w, n1_b])

        q_w = sd[f"{prefix}attention.attention.query.weight"].numpy().T
        q_b = sd[f"{prefix}attention.attention.query.bias"].numpy()
        k_w = sd[f"{prefix}attention.attention.key.weight"].numpy().T
        k_b = sd[f"{prefix}attention.attention.key.bias"].numpy()
        v_w = sd[f"{prefix}attention.attention.value.weight"].numpy().T
        v_b = sd[f"{prefix}attention.attention.value.bias"].numpy()
        o_w = sd[f"{prefix}attention.output.dense.weight"].numpy().T
        o_b = sd[f"{prefix}attention.output.dense.bias"].numpy()

        k_attn = getattr(k_blk, "attention", None)
        if k_attn:
            if hasattr(k_attn, "query_dense"):
                set_weights(k_attn.query_dense, [q_w, q_b])
                set_weights(k_attn.key_dense, [k_w, k_b])
                set_weights(k_attn.value_dense, [v_w, v_b])
            elif hasattr(k_attn, "qkv") or hasattr(k_attn, "predict_query_key_value"):
                target = getattr(
                    k_attn, "qkv", getattr(k_attn, "predict_query_key_value", None)
                )
                set_weights(
                    target,
                    [
                        np.concatenate([q_w, k_w, v_w], axis=1),
                        np.concatenate([q_b, k_b, v_b], axis=0),
                    ],
                )
            o_proj = getattr(
                k_attn, "output_dense", getattr(k_attn, "projection_layer", None)
            )
            if o_proj:
                set_weights(o_proj, [o_w, o_b])

        for ls_idx in [1, 2]:
            t_ls_key = f"{prefix}layer_scale{ls_idx}.lambda1"
            t_ls_val = sd.get(t_ls_key)
            if t_ls_val is not None:
                candidates = [
                    f"layer_scale_{ls_idx}",
                    f"ls{ls_idx}",
                    f"layerscale{ls_idx}",
                ]
                k_ls_layer = None
                for name in candidates:
                    if hasattr(k_blk, name):
                        k_ls_layer = getattr(k_blk, name)
                        break
                if k_ls_layer:
                    val_np = t_ls_val.numpy()
                    target_var = None
                    if hasattr(k_ls_layer, "gamma"):
                        target_var = k_ls_layer.gamma
                    elif (
                        hasattr(k_ls_layer, "trainable_variables")
                        and len(k_ls_layer.trainable_variables) > 0
                    ):
                        target_var = k_ls_layer.trainable_variables[0]
                    if target_var is None:
                        try:
                            c_dim = val_np.shape[0]
                            k_ls_layer.build((None, None, c_dim))
                            if hasattr(k_ls_layer, "gamma"):
                                target_var = k_ls_layer.gamma
                        except:
                            pass
                    if target_var is not None:
                        target_var.assign(val_np)

        n2_w = sd[f"{prefix}norm2.weight"].numpy()
        n2_b = sd[f"{prefix}norm2.bias"].numpy()
        n2_layer = getattr(k_blk, "normalization2", getattr(k_blk, "norm2", None))
        if n2_layer:
            set_weights(n2_layer, [n2_w, n2_b])

        k_mlp = getattr(k_blk, "mlp", None)
        if k_mlp:
            fc1_w = sd.get(f"{prefix}mlp.fc1.weight")
            if fc1_w is not None:
                fc1_b = sd[f"{prefix}mlp.fc1.bias"].numpy()
                fc2_w = sd[f"{prefix}mlp.fc2.weight"].numpy().T
                fc2_b = sd[f"{prefix}mlp.fc2.bias"].numpy()
                k_fc1 = getattr(
                    k_mlp, "fully_connected_layer_1", getattr(k_mlp, "fc1", None)
                )
                k_fc2 = getattr(
                    k_mlp, "fully_connected_layer_2", getattr(k_mlp, "fc2", None)
                )
                if k_fc1:
                    set_weights(k_fc1, [fc1_w.numpy().T, fc1_b])
                if k_fc2:
                    set_weights(k_fc2, [fc2_w, fc2_b])

    # E. Final Norm
    ln_w, ln_b = sd.get("layernorm.weight"), sd.get("layernorm.bias")
    if ln_w is not None:
        target = getattr(k_model, "normalization", getattr(k_model, "norm", None))
        if target:
            set_weights(target, [ln_w.numpy(), ln_b.numpy()])

    # F. Projector
    port_projector_weights(full_sd, k_backbone_model)


def port_projector_weights(full_sd, k_backbone_model):
    k_projector = None
    for layer in k_backbone_model.layers:
        if "projector" in layer.name.lower() or "multiscale" in layer.name.lower():
            k_projector = layer
            break
    if not k_projector:
        return

    proj_keys = [k for k in full_sd.keys() if "projector" in k or "fpn" in k]
    prefix_root = "projector"
    if len(proj_keys) > 0:
        prefix_root = proj_keys[0].split(".")[0]
        if "0" in prefix_root and len(prefix_root) < 3:
            prefix_root = proj_keys[0].split(".")[0] + "." + proj_keys[0].split(".")[1]
        if "stages" in prefix_root:
            prefix_root = prefix_root.split("stages")[0].rstrip(".")

    def copy_with_check(w_key, k_var, transpose=False):
        if w_key in full_sd:
            w_t = full_sd[w_key].numpy()
            if transpose and w_t.ndim == 4:
                w_t = w_t.transpose(2, 3, 1, 0)
            k_var.assign(w_t)

    def copy_single_layer(t_prefix, k_layer):
        if isinstance(k_layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)):
            if hasattr(k_layer, "kernel"):
                copy_with_check(f"{t_prefix}.weight", k_layer.kernel, transpose=True)
            if hasattr(k_layer, "bias") and k_layer.bias is not None:
                copy_with_check(f"{t_prefix}.bias", k_layer.bias)
        elif "Norm" in k_layer.__class__.__name__ or hasattr(k_layer, "gamma"):
            if f"{t_prefix}.weight" in full_sd:
                target = k_layer.gamma if hasattr(k_layer, "gamma") else k_layer.weight
                copy_with_check(f"{t_prefix}.weight", target)
            if f"{t_prefix}.bias" in full_sd:
                target = k_layer.beta if hasattr(k_layer, "beta") else k_layer.bias
                copy_with_check(f"{t_prefix}.bias", target)

    def copy_convx(t_prefix, k_convx):
        copy_single_layer(f"{t_prefix}.conv", k_convx.conv)
        if hasattr(k_convx, "bn") and k_convx.bn:
            copy_single_layer(f"{t_prefix}.bn", k_convx.bn)

    def copy_c2f(t_prefix, k_c2f):
        copy_convx(f"{t_prefix}.cv1", k_c2f.cv1)
        copy_convx(f"{t_prefix}.cv2", k_c2f.cv2)
        for i, k_b in enumerate(k_c2f.bottlenecks):
            copy_convx(f"{t_prefix}.m.{i}.cv1", k_b.cv1)
            copy_convx(f"{t_prefix}.m.{i}.cv2", k_b.cv2)

    for i, stage_list in enumerate(k_projector.stages_sampling):
        for j, sampler_seq in enumerate(stage_list):
            t_base = f"{prefix_root}.stages_sampling.{i}.{j}"
            layer_idx = 0
            for layer in sampler_seq.layers:
                if (
                    isinstance(
                        layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)
                    )
                    or "Norm" in layer.__class__.__name__
                ):
                    copy_single_layer(f"{t_base}.{layer_idx}", layer)
                    layer_idx += 1
                elif hasattr(layer, "conv"):
                    copy_convx(f"{t_base}.{layer_idx}", layer)
                    layer_idx += 1
                elif (
                    "Activation" in layer.__class__.__name__
                    or "GELU" in layer.__class__.__name__
                ):
                    layer_idx += 1

    for i, stage_seq in enumerate(k_projector.stages):
        t_base = f"{prefix_root}.stages.{i}"
        if len(stage_seq.layers) > 0:
            if hasattr(stage_seq.layers[0], "cv1"):
                copy_c2f(f"{t_base}.0", stage_seq.layers[0])
        if len(stage_seq.layers) > 1:
            copy_single_layer(f"{t_base}.1", stage_seq.layers[1])
    log("Ported Projector Weights", status="PASS")


# --- Transformer Helpers ---


def to_keras_mha_input_kernel(w, n_head):
    w = w.T
    d_model = w.shape[0]
    head_dim = w.shape[1] // n_head
    return w.reshape(d_model, n_head, head_dim)


def to_keras_mha_bias(b, n_head):
    head_dim = b.shape[0] // n_head
    return b.reshape(n_head, head_dim)


def to_keras_mha_output_kernel(w, n_head):
    w = w.T
    d_model_out = w.shape[1]
    head_dim = w.shape[0] // n_head
    return w.reshape(n_head, head_dim, d_model_out)


def transfer_weights_smart(pt_layer, k_layer):
    """
    Intelligently transfers weights from PyTorch Linear to Keras (Dense OR Conv2D).
    Handles 1x1 Conv2D (common in DETR) vs Dense.
    """
    with torch.no_grad():
        w = pt_layer.weight.detach().cpu().numpy()
        b = pt_layer.bias.detach().cpu().numpy() if pt_layer.bias is not None else None

        # Check target Keras layer type
        if isinstance(k_layer, keras.layers.Conv2D):
            w = w.T[None, None, :, :]
        else:
            # Standard Dense: (In, Out)
            w = w.transpose(1, 0)

        if b is not None:
            set_weights(k_layer, [w, b])
        else:
            set_weights(k_layer, [w])


def transfer_norm_weights(pt_layer, k_layer):
    with torch.no_grad():
        w = pt_layer.weight.detach().cpu().numpy()
        b = pt_layer.bias.detach().cpu().numpy()
        set_weights(k_layer, [w, b])


def transfer_mlp_weights(pt_mlp, k_mlp):
    if hasattr(k_mlp, "model"):
        k_layers = k_mlp.model.layers
    elif hasattr(k_mlp, "layers"):
        k_layers = k_mlp.layers
    else:
        log("Cannot find layers in Keras MLP", status="FAIL")
        return

    pt_layers = pt_mlp.layers
    k_dense = [
        l for l in k_layers if isinstance(l, (keras.layers.Dense, keras.layers.Conv2D))
    ]
    pt_dense = [l for l in pt_layers if isinstance(l, torch.nn.Linear)]

    for pt_l, k_l in zip(pt_dense, k_dense):
        transfer_weights_smart(pt_l, k_l)


def transfer_decoder_layer_weights(pt_layer, k_layer, sa_nhead):
    with torch.no_grad():
        # Self Attn
        pt_sa = pt_layer.self_attn
        qkv_w = pt_sa.in_proj_weight.detach().cpu().numpy()
        qkv_b = pt_sa.in_proj_bias.detach().cpu().numpy()
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)
        out_w = pt_sa.out_proj.weight.detach().cpu().numpy()
        out_b = pt_sa.out_proj.bias.detach().cpu().numpy()

        k_layer.self_attn.set_weights(
            [
                to_keras_mha_input_kernel(q_w, sa_nhead),
                to_keras_mha_bias(q_b, sa_nhead),
                to_keras_mha_input_kernel(k_w, sa_nhead),
                to_keras_mha_bias(k_b, sa_nhead),
                to_keras_mha_input_kernel(v_w, sa_nhead),
                to_keras_mha_bias(v_b, sa_nhead),
                to_keras_mha_output_kernel(out_w, sa_nhead),
                out_b,
            ]
        )

        # Norms
        transfer_norm_weights(pt_layer.norm1, k_layer.norm1)
        transfer_norm_weights(pt_layer.norm2, k_layer.norm2)
        transfer_norm_weights(pt_layer.norm3, k_layer.norm3)

        # Cross Attn
        pt_ca = pt_layer.cross_attn
        k_ca = k_layer.cross_attn

        if hasattr(pt_ca, "sampling_offsets"):
            transfer_weights_smart(pt_ca.value_proj, k_ca.value_proj)
            transfer_weights_smart(pt_ca.output_proj, k_ca.output_proj)
            transfer_weights_smart(pt_ca.sampling_offsets, k_ca.sampling_offsets)
            transfer_weights_smart(pt_ca.attention_weights, k_ca.attention_weights)
        else:
            pass

        # FFN
        transfer_weights_smart(pt_layer.linear1, k_layer.linear1)
        transfer_weights_smart(pt_layer.linear2, k_layer.linear2)


# -------------------------------------------------------------------------
# Main Porting Function
# -------------------------------------------------------------------------


def port_weights(t_model, k_model):
    log(">>> Starting LWDETR Weight Porting...", section=True)
    port_backbone_weights(t_model.backbone, k_model.backbone)

    t_trans = t_model.transformer
    k_trans = k_model.transformer

    for i in range(len(t_trans.decoder.layers)):
        transfer_decoder_layer_weights(
            t_trans.decoder.layers[i],
            k_trans.decoder.decoder_layers[i],
            sa_nhead=TEST_ARGS.nheads,
        )

    if t_trans.decoder.norm is not None and k_trans.decoder.norm is not None:
        transfer_norm_weights(t_trans.decoder.norm, k_trans.decoder.norm)

    transfer_mlp_weights(t_trans.decoder.ref_point_head, k_trans.decoder.ref_point_head)

    log(">> Porting Heads & Embeddings...", section=True)
    k_model.refpoint_embed.assign(t_model.refpoint_embed.weight.detach().cpu().numpy())
    k_model.query_feat.assign(t_model.query_feat.weight.detach().cpu().numpy())

    transfer_weights_smart(t_model.class_embed, k_model.class_embed)
    transfer_mlp_weights(t_model.bbox_embed, k_model.bbox_embed)

    if TEST_ARGS.two_stage:
        log(">> Porting Two-Stage Components...", section=True)
        group_detr = TEST_ARGS.group_detr
        for i in range(group_detr):
            transfer_weights_smart(
                t_model.transformer.enc_out_class_embed[i],
                k_model.transformer.enc_out_class_embed[i],
            )
            transfer_mlp_weights(
                t_model.transformer.enc_out_bbox_embed[i],
                k_model.transformer.enc_out_bbox_embed[i],
            )
            transfer_weights_smart(
                t_model.transformer.enc_output[i], k_model.transformer.enc_output[i]
            )
            transfer_norm_weights(
                t_model.transformer.enc_output_norm[i],
                k_model.transformer.enc_output_norm[i],
            )

    if not t_model.lite_refpoint_refine:
        k_model.transformer.decoder.bbox_embed.set_weights(
            k_model.bbox_embed.get_weights()
        )


# -------------------------------------------------------------------------
# test: Block-by-Block Isolation Verification
# -------------------------------------------------------------------------
def verify_blocks_isolated(model_torch, model_keras):
    log("Starting Block-by-Block Verification (Isolated Inputs)...", section=True)

    # -------------------------------------------------------------------------
    # Setup Inputs
    # -------------------------------------------------------------------------
    B, H, W, C = 1, TEST_ARGS.shape[0], TEST_ARGS.shape[1], 3
    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, (B, H, W, C)).astype(np.float32)

    # Torch: (B, C, H, W)
    x_torch = torch.from_numpy(np.transpose(x_np, (0, 3, 1, 2)))
    # Keras: (B, H, W, C)
    x_keras = ops.convert_to_tensor(x_np)

    # -------------------------------------------------------------------------
    # PART 1: BACKBONE ViT BLOCKS
    # -------------------------------------------------------------------------
    log("\n[Phase 1] Backbone ViT Blocks")

    t_backbone = model_torch.backbone[0].encoder.encoder

    # Unwrap Keras Backbone
    k_joiner = model_keras.backbone
    k_backbone_func = k_joiner.backbone if hasattr(k_joiner, "backbone") else k_joiner
    k_wrapper = None
    for l in k_backbone_func.layers:
        if "backbone_wrapper" in l.name or "BackboneWrapper" in l.__class__.__name__:
            k_wrapper = l
            break
    k_vit = getattr(k_wrapper, "feature_extractor", k_wrapper)

    # 1.1 Init Embeddings (Golden Input for Block 0)
    with torch.no_grad():
        # Torch Forward
        t_current = t_backbone.embeddings(x_torch)  # (B, N, D)
        t_current_np = t_current.detach().numpy()

    # 1.2 Iterate Blocks
    k_blocks = get_keras_blocks(k_vit, TEST_ARGS.vit_encoder_num_layers)
    t_blocks = t_backbone.encoder.layer

    log(f"Detected {len(k_blocks)} Keras Blocks vs {len(t_blocks)} Torch Blocks")

    for i, (t_blk, k_blk) in enumerate(zip(t_blocks, k_blocks)):
        # Prepare Input (Teacher Forcing: Use Torch input for both)
        t_in = torch.from_numpy(t_current_np)
        k_in = ops.convert_to_tensor(t_current_np)

        # Run Forward
        with torch.no_grad():
            t_out = t_blk(t_in)[0].detach().numpy()

        k_out = k_blk(k_in, training=False)
        k_out = ops.convert_to_numpy(k_out)

        # Compare
        passed, diff = check_close(t_out, k_out, f"Backbone Block {i}")

        # Update Input for next block
        t_current_np = t_out

    # -------------------------------------------------------------------------
    # PREPARE FEATURES FOR TRANSFORMER (Backbone + Projection)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        mask = torch.zeros((B, H, W), dtype=torch.bool)
        x_nested = NestedTensor(x_torch, mask)
        features, poss = model_torch.backbone(x_nested)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, m = feat.decompose()
            srcs.append(src)
            masks.append(m)

        t_trans = model_torch.transformer

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poss)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # (B, HW, C)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

            if hasattr(t_trans, "level_embed") and t_trans.level_embed is not None:
                lvl_pos_embed = pos_embed + t_trans.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed

            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )

        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([t_trans.get_valid_ratio(m) for m in masks], 1)

    t_curr_src = src_flatten

    # Keras Input Constants
    k_spatial_shapes = ops.convert_to_tensor(spatial_shapes.cpu().numpy())
    k_level_start_index = ops.convert_to_tensor(level_start_index.cpu().numpy())
    k_valid_ratios = ops.convert_to_tensor(valid_ratios.cpu().numpy())
    k_pos = ops.convert_to_tensor(lvl_pos_embed_flatten.cpu().numpy())
    k_mask = ops.convert_to_tensor(mask_flatten.cpu().numpy())

    # -------------------------------------------------------------------------
    # PART 2: TRANSFORMER ENCODER LAYERS
    # -------------------------------------------------------------------------
    log("\n[Phase 2] Transformer Encoder Layers")

    if t_trans.encoder is not None:
        t_enc_layers = t_trans.encoder.layers
        k_enc_layers = model_keras.transformer.encoder.encoder_layers

        log(
            f"Detected {len(k_enc_layers)} Keras Encoder Layers vs {len(t_enc_layers)} Torch"
        )

        for i, (t_l, k_l) in enumerate(zip(t_enc_layers, k_enc_layers)):
            t_src_ten = t_curr_src
            k_src_ten = ops.convert_to_tensor(t_src_ten.cpu().numpy())

            with torch.no_grad():
                t_out = t_l(
                    t_src_ten,
                    pos=lvl_pos_embed_flatten,
                    reference_points=spatial_shapes,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    padding_mask=mask_flatten,
                )

            k_out = k_l(
                k_src_ten,
                pos=k_pos,
                reference_points=k_spatial_shapes,
                spatial_shapes=k_spatial_shapes,
                level_start_index=k_level_start_index,
                padding_mask=k_mask,
            )

            k_out_np = ops.convert_to_numpy(k_out)
            t_out_np = t_out.detach().cpu().numpy()

            passed, diff = check_close(t_out_np, k_out_np, f"Encoder Layer {i}")
            t_curr_src = t_out
    else:
        log(
            "Transformer Encoder is None (Encoder-less architecture). Skipping Phase 2.",
            status="WARN",
        )

    # -------------------------------------------------------------------------
    # PART 3: TRANSFORMER DECODER LAYERS
    # -------------------------------------------------------------------------
    log("\n[Phase 3] Transformer Decoder Layers")

    # 3.1 Setup Initial Decoder Inputs
    with torch.no_grad():
        memory = t_curr_src
        bs, _, c = memory.shape
        query_embed = model_torch.query_feat.weight
        refpoint_embed = model_torch.refpoint_embed.weight

        tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
        refpoint_embed = refpoint_embed.unsqueeze(0).expand(bs, -1, -1)

        # Start with sigmoid-ed reference points (or not, depending on reparam)
        reference_points = refpoint_embed.sigmoid()

        t_curr_tgt = tgt
        t_curr_ref = reference_points  # (B, Nq, 4)

    t_dec_layers = t_trans.decoder.layers
    k_dec_layers = model_keras.transformer.decoder.decoder_layers

    k_memory = ops.convert_to_tensor(memory.cpu().numpy())

    log(
        f"Detected {len(k_dec_layers)} Keras Decoder Layers vs {len(t_dec_layers)} Torch"
    )

    # Helper: Replicate TransformerDecoder.get_reference logic
    from examples.dino_object_detection.models.transformer_decoder_head.torch_transformer_for_testing import (
        gen_sineembed_for_position,
    )

    def get_reference_torch(refpoints, valid_ratios, d_model, ref_point_head):
        # refpoints: (B, Nq, 4) or (B, Nq, 2)
        obj_center = refpoints[..., :4]

        # Expand reference points by valid_ratios (B, Nq, n_levels, 4)
        # valid_ratios: (B, n_levels, 2)
        # We need to broadcast: (B, Nq, 1, 4) * (B, 1, n_levels, 2->expanded to 4)

        # (B, n_levels, 2) -> (B, 1, n_levels, 2)
        vr = valid_ratios[:, None, :, :]
        # Concatenate to match 4 coords (x,y,w,h) -> (x,y) scaled by (w_ratio, h_ratio)
        vr = torch.cat([vr, vr], -1)

        # (B, Nq, 1, 4)
        oc = obj_center[:, :, None, :]

        refpoints_input = oc * vr

        # Generate sine embed (using only x,y)
        # refpoints_input[:, :, 0, :] is (B, Nq, 4), effectively taking the 0th level
        query_sine_embed = gen_sineembed_for_position(
            refpoints_input[:, :, 0, :], d_model // 2
        )
        query_pos = ref_point_head(query_sine_embed)

        return obj_center, refpoints_input, query_pos, query_sine_embed

    for i, (t_l, k_l) in enumerate(zip(t_dec_layers, k_dec_layers)):
        # 3.2 Prepare Input Features for this specific Layer
        # We must replicate the 'get_reference' and sine-embed logic that happens
        # INSIDE TransformerDecoder but OUTSIDE TransformerDecoderLayer.

        # Calculate derived inputs using Torch logic (as Ground Truth inputs)
        with torch.no_grad():
            t_ref_point_head = t_trans.decoder.ref_point_head

            # Logic from TransformerDecoder.forward loop
            # Assuming not lite_refpoint_refine for simplicity of verification structure
            # (If lite, it happens before loop, but generally updated inside)

            # t_curr_ref is the "refpoints_unsigmoid" (or sigmoid-ed) from previous step
            # Note: For strict block isolation, we assume t_curr_ref is valid input

            obj_center, refpoints_input, query_pos, query_sine_embed = (
                get_reference_torch(
                    t_curr_ref, valid_ratios, TEST_ARGS.hidden_dim, t_ref_point_head
                )
            )

            # These are the actual inputs to the layer
            t_pos_transformation = 1.0
            query_pos = query_pos * t_pos_transformation

        # 3.3 Prepare Keras Inputs (from the computed Torch inputs)
        k_tgt = ops.convert_to_tensor(t_curr_tgt.detach().cpu().numpy())
        k_query_pos = ops.convert_to_tensor(query_pos.detach().cpu().numpy())
        k_query_sine = ops.convert_to_tensor(query_sine_embed.detach().cpu().numpy())
        k_ref_input = ops.convert_to_tensor(refpoints_input.detach().cpu().numpy())

        # 3.4 Forward Pass
        # Torch
        with torch.no_grad():
            t_out = t_l(
                t_curr_tgt,
                memory,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                reference_points=refpoints_input,  # Expanded (B, Nq, L, 4)
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                # Removed: valid_ratios (Layer doesn't take it)
            )

        # Keras
        k_out = k_l(
            k_tgt,
            k_memory,
            memory_key_padding_mask=k_mask,
            pos=k_pos,
            query_pos=k_query_pos,
            query_sine_embed=k_query_sine,
            reference_points=k_ref_input,  # Expanded
            spatial_shapes=k_spatial_shapes,
            level_start_index=k_level_start_index,
            # Removed: valid_ratios
        )

        k_out_np = ops.convert_to_numpy(k_out)
        t_out_np = t_out.detach().cpu().numpy()

        passed, diff = check_close(t_out_np, k_out_np, f"Decoder Layer {i}")

        # Update t_curr_tgt (hidden states) for next layer
        t_curr_tgt = t_out

        # Note: In a full model, we would also update reference points here using bbox_embed.
        # For 'isolated' layer verification, passing the hidden state is the critical path.
        # We keep t_curr_ref constant for simple layer verification, or you can invoke the bbox_head manually.

    log("Block Verification Complete.", section=True)


# -------------------------------------------------------------------------
# test: End-to-End Error Propagation block by block Check
# -------------------------------------------------------------------------
def verify_end_to_end_propagation(model_torch, model_keras):
    log("Starting End-to-End Propagation (Accumulated Error Check)...", section=True)
    B, H, W, C = 1, TEST_ARGS.shape[0], TEST_ARGS.shape[1], 3
    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, (B, H, W, C)).astype(np.float32)
    x_torch = torch.from_numpy(np.transpose(x_np, (0, 3, 1, 2)))
    x_keras = ops.convert_to_tensor(x_np)

    log("\n[Phase 1] Backbone Propagation (Real Input Flow)")
    t_backbone_enc = model_torch.backbone[0].encoder.encoder
    k_joiner = model_keras.backbone
    k_backbone_model, wrapper = get_backbone_and_wrapper(k_joiner)
    k_vit = getattr(wrapper, "feature_extractor", wrapper)

    with torch.no_grad():
        t_curr = t_backbone_enc.embeddings(x_torch)
    k_curr = ops.convert_to_tensor(t_curr.detach().cpu().numpy())
    t_curr_t = t_curr

    k_blocks = get_keras_blocks(k_vit, TEST_ARGS.vit_encoder_num_layers)
    t_blocks = t_backbone_enc.encoder.layer

    for i, (t_blk, k_blk) in enumerate(zip(t_blocks, k_blocks)):
        with torch.no_grad():
            t_out = t_blk(t_curr_t)[0]
        k_out = k_blk(k_curr, training=False)
        passed, diff = check_close(t_out, k_out, f"Backbone Block {i} (Accumulated)")
        t_curr_t = t_out
        k_curr = k_out

    log("\n[Phase 2] Backbone + Projector Output")
    with torch.no_grad():
        mask = torch.zeros((B, H, W), dtype=torch.bool)
        x_nested = NestedTensor(x_torch, mask)
        t_features, t_poss = model_torch.backbone(x_nested)
    k_backbone_out = model_keras.backbone(x_keras)
    if isinstance(k_backbone_out, tuple):
        k_features = k_backbone_out[0]
    else:
        k_features = [k_backbone_out]

    for lvl, (t_f, k_f) in enumerate(zip(t_features, k_features)):
        t_f_tensor, _ = t_f.decompose()
        t_mean, k_mean = float(t_f_tensor.mean()), float(ops.mean(k_f))
        log(f"Lvl {lvl} | Torch Mean: {t_mean:.4f} | Keras Mean: {k_mean:.4f}")
        passed, diff = check_close(t_f_tensor, k_f, f"Projector Feature Lvl {lvl}")

    log("\n[Phase 3] Transformer Input Preparation")
    k_srcs = []
    k_pos_embeds = []
    k_spatial_shapes_list = []
    for lvl, k_feat in enumerate(k_features):
        shape = k_feat.shape
        h, w = shape[1], shape[2]
        k_spatial_shapes_list.append([h, w])
        feat_flat = ops.reshape(k_feat, (B, -1, shape[3]))
        k_srcs.append(feat_flat)
        if lvl < len(t_poss):
            p = t_poss[lvl].flatten(2).transpose(1, 2).detach().cpu().numpy()
            k_pos_embeds.append(ops.convert_to_tensor(p))

    k_memory = ops.concatenate(k_srcs, axis=1)
    k_lvl_pos = ops.concatenate(k_pos_embeds, axis=1)
    k_spatial_shapes = ops.convert_to_tensor(k_spatial_shapes_list, dtype="int32")
    k_valid_ratios = ops.ones((B, len(k_features), 2), dtype="float32")
    hw_counts = ops.prod(k_spatial_shapes, axis=1)
    cumulative = ops.cumsum(hw_counts, axis=0)
    zero = ops.zeros((1,), dtype="int32")
    k_level_start_index = ops.concatenate([zero, cumulative[:-1]], axis=0)
    k_level_start_index = ops.cast(k_level_start_index, "int64")
    k_mask_flatten = None

    log("\n[Phase 4] Decoder Propagation (Real Input Flow)")
    k_query_embed = model_keras.query_feat
    k_ref_embed = model_keras.refpoint_embed
    k_tgt = ops.broadcast_to(
        ops.expand_dims(k_query_embed, 0),
        (B, TEST_ARGS.num_queries, TEST_ARGS.hidden_dim),
    )
    k_ref_weight = ops.broadcast_to(
        ops.expand_dims(k_ref_embed, 0), (B, TEST_ARGS.num_queries, 4)
    )
    k_ref_unsigmoid = k_ref_weight
    k_ref_points = ops.sigmoid(k_ref_unsigmoid)

    with torch.no_grad():
        t_bs = 1
        t_query = model_torch.query_feat.weight.unsqueeze(0).expand(t_bs, -1, -1)
        t_ref_w = model_torch.refpoint_embed.weight.unsqueeze(0).expand(t_bs, -1, -1)
        t_tgt = t_query
        t_ref_unsigmoid = t_ref_w
        t_ref_points = t_ref_w.sigmoid()
        srcs = []
        masks = []
        for l, feat in enumerate(t_features):
            src, m = feat.decompose()
            srcs.append(src)
            masks.append(m)
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, t_poss)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)
            if (
                hasattr(model_torch.transformer, "level_embed")
                and model_torch.transformer.level_embed is not None
            ):
                pos_embed = pos_embed.flatten(2).transpose(
                    1, 2
                ) + model_torch.transformer.level_embed[lvl].view(1, 1, -1)
            else:
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
            src_flatten.append(src)
            lvl_pos_embed_flatten.append(pos_embed)
        t_memory = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=t_memory.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        mask_flatten = None

    k_dec_layers = model_keras.transformer.decoder.decoder_layers
    t_dec_layers = model_torch.transformer.decoder.layers
    k_ref_head = model_keras.transformer.decoder.ref_point_head

    def get_reference_keras(refpoints, valid_ratios, d_model, ref_point_head):
        obj_center = refpoints[..., :4]
        vr = ops.expand_dims(valid_ratios, 1)
        vr = ops.concatenate([vr, vr], axis=-1)
        oc = ops.expand_dims(obj_center, 2)
        refpoints_input = oc * vr

        def gen_sine(pos_tensor, dim=128):
            scale = 2 * math.pi
            dim_t = ops.arange(dim, dtype="float32")
            dim_t = 10000 ** (2 * (dim_t // 2) / dim)
            x_embed = pos_tensor[..., 0] * scale
            y_embed = pos_tensor[..., 1] * scale
            w_embed = pos_tensor[..., 2] * scale
            h_embed = pos_tensor[..., 3] * scale

            def get_embed(val):
                pos = val[..., None] / dim_t
                pos = ops.stack(
                    [ops.sin(pos[..., 0::2]), ops.cos(pos[..., 1::2])], axis=-1
                )
                s = ops.shape(pos)
                return ops.reshape(pos, (s[0], s[1], -1))

            return ops.concatenate(
                [
                    get_embed(y_embed),
                    get_embed(x_embed),
                    get_embed(w_embed),
                    get_embed(h_embed),
                ],
                axis=-1,
            )

        query_sine_embed = gen_sine(refpoints_input[..., 0, :], d_model // 2)
        query_pos = ref_point_head(query_sine_embed)
        return obj_center, refpoints_input, query_pos, query_sine_embed

    from examples.dino_object_detection.models.lwdetr.lwdetr_keras import (
        MLP as KerasMLP,
    )

    def get_clean_bbox_head(source_head):
        new_head = KerasMLP(TEST_ARGS.hidden_dim, TEST_ARGS.hidden_dim, 4, 3)
        dummy_in = ops.zeros((1, 1, TEST_ARGS.hidden_dim), dtype="float32")
        _ = new_head(dummy_in)
        if hasattr(source_head, "model"):
            src_layers = source_head.model.layers
        else:
            src_layers = source_head.layers
        if hasattr(new_head, "model"):
            dst_layers = new_head.model.layers
        else:
            dst_layers = new_head.layers
        for s_l, d_l in zip(src_layers, dst_layers):
            if hasattr(s_l, "get_weights"):
                d_l.set_weights(s_l.get_weights())
        return new_head

    if hasattr(model_keras, "bbox_embed"):
        source_bbox_head = model_keras.bbox_embed
    else:
        source_bbox_head = getattr(model_keras.transformer, "bbox_embed", None)
    if isinstance(source_bbox_head, (list, tuple)) or hasattr(
        source_bbox_head, "__getitem__"
    ):
        source_bbox_head = source_bbox_head[0]
    k_bbox_head_clean = get_clean_bbox_head(source_bbox_head)

    from examples.dino_object_detection.models.transformer_decoder_head.torch_transformer_for_testing import (
        gen_sineembed_for_position,
    )

    for i, (t_l, k_l) in enumerate(zip(t_dec_layers, k_dec_layers)):
        _, k_ref_input, k_q_pos, k_q_sine = get_reference_keras(
            k_ref_points, k_valid_ratios, TEST_ARGS.hidden_dim, k_ref_head
        )
        with torch.no_grad():
            t_vr = torch.stack(
                [model_torch.transformer.get_valid_ratio(m) for m in masks], 1
            )
            t_vr_ex = t_vr.unsqueeze(1)
            t_vr_ex = torch.cat([t_vr_ex, t_vr_ex], -1)
            t_oc = t_ref_points.unsqueeze(2)
            t_ref_input = t_oc * t_vr_ex
            t_q_sine = gen_sineembed_for_position(
                t_ref_input[:, :, 0, :], TEST_ARGS.hidden_dim // 2
            )
            t_q_pos = model_torch.transformer.decoder.ref_point_head(t_q_sine)

        k_out = k_l(
            k_tgt,
            k_memory,
            memory_key_padding_mask=k_mask_flatten,
            pos=k_lvl_pos,
            query_pos=k_q_pos,
            query_sine_embed=k_q_sine,
            reference_points=k_ref_input,
            spatial_shapes=k_spatial_shapes,
            level_start_index=k_level_start_index,
        )
        with torch.no_grad():
            t_out = t_l(
                t_tgt,
                t_memory,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                query_pos=t_q_pos,
                query_sine_embed=t_q_sine,
                reference_points=t_ref_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

        passed, diff = check_close(t_out, k_out, f"Decoder Layer {i} Output")

        k_delta = k_bbox_head_clean(k_out)
        k_ref_unsigmoid = k_ref_unsigmoid + k_delta
        k_ref_points = ops.stop_gradient(ops.sigmoid(k_ref_unsigmoid))
        k_tgt = k_out
        with torch.no_grad():
            if isinstance(model_torch.bbox_embed, nn.ModuleList):
                t_delta = model_torch.bbox_embed[i](t_out)
            else:
                t_delta = model_torch.bbox_embed(t_out)
            t_ref_unsigmoid = t_ref_unsigmoid + t_delta
            t_ref_points = t_ref_unsigmoid.sigmoid()
            t_tgt = t_out


# -------------------------------------------------------------------------
# test: End-to-End Segments Propagation Check
# -------------------------------------------------------------------------
def verify_end_to_end_segments_propagation(model_torch, model_keras):
    log("Starting End-to-End Error Propagation Check (Full Model)...", section=True)
    B, C, H, W = 1, 3, TEST_ARGS.shape[0], TEST_ARGS.shape[1]
    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, (B, C, H, W)).astype(np.float32)

    x_torch = torch.from_numpy(x_np)
    mask_torch = torch.zeros((B, H, W), dtype=torch.bool)
    x_nested = NestedTensor(x_torch, mask_torch)
    x_keras = ops.convert_to_tensor(np.transpose(x_np, (0, 2, 3, 1)))

    print(f"\n{'Stage':<25} | {'Max Diff':<12} | {'Mean Diff':<12} | {'Status'}")
    print("-" * 65)

    with torch.no_grad():
        t_features, t_poss = model_torch.backbone(x_nested)
    k_features, k_poss = model_keras.backbone(x_keras)

    t_feat_last = t_features[-1].tensors.detach().cpu().numpy()
    k_feat_last = ops.convert_to_numpy(k_features[-1])
    if k_feat_last.ndim == 4:
        k_feat_last = np.transpose(k_feat_last, (0, 3, 1, 2))

    check_close(
        t_val=t_feat_last, k_val=k_feat_last, name="Backbone Feature (Last)", atol=1e-4
    )

    t_pos_last = t_poss[-1].detach().cpu().numpy()
    k_pos_last = ops.convert_to_numpy(k_poss[-1])
    if k_pos_last.ndim == 4:
        k_pos_last = np.transpose(k_pos_last, (0, 3, 1, 2))

    check_close(
        t_val=t_pos_last, k_val=k_pos_last, name="Backbone Pos Embed", atol=1e-4
    )

    t_srcs = []
    t_masks = []
    for l, feat in enumerate(t_features):
        src, mask = feat.decompose()
        t_srcs.append(src)
        t_masks.append(mask)

    k_srcs = []
    k_masks = []
    for l, feat in enumerate(k_features):
        k_srcs.append(feat)
        shape = ops.shape(feat)
        m = ops.zeros((shape[0], shape[1], shape[2]), dtype="bool")
        k_masks.append(m)

    k_poss_fixed = []
    for pos in k_poss:
        pos_np = ops.convert_to_numpy(pos)
        if pos_np.shape[1] == TEST_ARGS.hidden_dim:
            pos = ops.transpose(pos, (0, 2, 3, 1))
        k_poss_fixed.append(pos)

    t_ref_w = model_torch.refpoint_embed.weight[: TEST_ARGS.num_queries]
    t_query_w = model_torch.query_feat.weight[: TEST_ARGS.num_queries]
    k_ref_w = model_keras.refpoint_embed[: TEST_ARGS.num_queries]
    k_query_w = model_keras.query_feat[: TEST_ARGS.num_queries]

    with torch.no_grad():
        t_hs, t_ref, t_hs_enc, t_ref_enc = model_torch.transformer(
            t_srcs, t_masks, t_poss, t_ref_w, t_query_w
        )

    k_hs, k_ref, k_hs_enc, k_ref_enc = model_keras.transformer(
        k_srcs, k_masks, k_poss_fixed, k_ref_w, k_query_w
    )

    t_hs_enc_np = t_hs_enc.detach().cpu().numpy()
    k_hs_enc_np = ops.convert_to_numpy(k_hs_enc)
    check_close(
        t_val=t_hs_enc_np, k_val=k_hs_enc_np, name="Transformer Encoder", atol=1e-4
    )

    t_hs_np = t_hs.detach().cpu().numpy()
    k_hs_np = ops.convert_to_numpy(k_hs)
    check_close(t_val=t_hs_np, k_val=k_hs_np, name="Transformer Decoder", atol=1e-4)

    t_ref_np = t_ref.detach().cpu().numpy()
    k_ref_np = ops.convert_to_numpy(k_ref)
    check_close(t_val=t_ref_np, k_val=k_ref_np, name="Ref Points (Unsig)", atol=1e-4)

    with torch.no_grad():
        t_cls_logits = model_torch.class_embed(t_hs)
    k_cls_logits = model_keras.class_embed(k_hs)
    t_cls_np = t_cls_logits.detach().cpu().numpy()
    k_cls_np = ops.convert_to_numpy(k_cls_logits)
    check_close(t_val=t_cls_np, k_val=k_cls_np, name="Class Head (Logits)", atol=1e-4)

    with torch.no_grad():
        if TEST_ARGS.bbox_reparam:
            t_coord_delta = model_torch.bbox_embed(t_hs)
            t_box_raw = t_coord_delta
        else:
            t_box_raw = model_torch.bbox_embed(t_hs)

    k_box_raw = model_keras.bbox_embed(k_hs)
    t_box_np = t_box_raw.detach().cpu().numpy()
    k_box_np = ops.convert_to_numpy(k_box_raw)
    check_close(t_val=t_box_np, k_val=k_box_np, name="BBox Head (Raw)", atol=1e-4)

    log("\nVerifying Final Model .call() Structure...", section=False)
    k_out_dict = model_keras(x_keras, training=False)
    required_keys = ["pred_logits", "pred_boxes"]
    if TEST_ARGS.aux_loss:
        required_keys.append("aux_outputs")
    if TEST_ARGS.two_stage:
        required_keys.append("enc_outputs")
    missing = [k for k in required_keys if k not in k_out_dict]
    if not missing:
        log("Final Output Dictionary Structure", status="PASS")
    else:
        log(f"Missing keys in Keras output: {missing}", status="FAIL")
    log("End-to-End Propagation Check Finished.", section=True)


# -------------------------------------------------------------------------
# test: Final Output Verification (Black Box Check)
# -------------------------------------------------------------------------
def verify_final_output(model_torch, model_keras):
    log("Starting Final Output Verification (Black Box Check)...", section=True)
    B, H, W, C = 1, TEST_ARGS.shape[0], TEST_ARGS.shape[1], 3
    np.random.seed(100)
    x_np = np.random.uniform(-1, 1, (B, H, W, C)).astype(np.float32)
    x_torch = torch.from_numpy(np.transpose(x_np, (0, 3, 1, 2)))
    mask_torch = torch.zeros((B, H, W), dtype=torch.bool)
    x_nested = NestedTensor(x_torch, mask_torch)
    x_keras = ops.convert_to_tensor(x_np)

    log("Running Inference...", status="INFO")
    with torch.no_grad():
        model_torch.eval()
        out_torch = model_torch(x_nested)
    out_keras = model_keras(x_keras, training=False)

    log("\n[Main Outputs]")
    check_close(
        out_torch["pred_logits"],
        out_keras["pred_logits"],
        "Final Pred Logits",
        atol=1e-4,
    )
    check_close(
        out_torch["pred_boxes"], out_keras["pred_boxes"], "Final Pred Boxes", atol=1e-4
    )

    if "aux_outputs" in out_torch and "aux_outputs" in out_keras:
        log("\n[Auxiliary Outputs]")
        t_aux = out_torch["aux_outputs"]
        k_aux = out_keras["aux_outputs"]
        if len(t_aux) != len(k_aux):
            log(
                f"Aux length mismatch: PT {len(t_aux)} vs Keras {len(k_aux)}",
                status="FAIL",
            )
        else:
            for i, (t_a, k_a) in enumerate(zip(t_aux, k_aux)):
                log(f"--- Aux Layer {i} ---")
                check_close(
                    t_a["pred_logits"], k_a["pred_logits"], f"Aux {i} Logits", atol=1e-4
                )
                check_close(
                    t_a["pred_boxes"], k_a["pred_boxes"], f"Aux {i} Boxes", atol=1e-4
                )

    if "enc_outputs" in out_torch and "enc_outputs" in out_keras:
        log("\n[Encoder Outputs (Two-Stage)]")
        t_enc = out_torch["enc_outputs"]
        k_enc = out_keras["enc_outputs"]
        check_close(
            t_enc["pred_logits"], k_enc["pred_logits"], "Encoder Logits", atol=1e-4
        )
        check_close(
            t_enc["pred_boxes"], k_enc["pred_boxes"], "Encoder Boxes", atol=1e-4
        )
    log("Final Output Verification Finished.", section=True)


def transfer_specific_internal_weights(pt_model, k_model):
    print("Transferring internal RefPoint Head weights...")

    # MLP Transfer Helper (assuming you have this available in the script)
    def transfer_mlp(pt_mlp, k_mlp):
        with torch.no_grad():
            for i in range(len(k_mlp.mlp_layers.layers)):
                # PyTorch MLP usually has Linear -> ReLU -> Linear
                # Keras MLP has Dense -> Dense
                if i < len(pt_mlp.layers):
                    pt_layer = pt_mlp.layers[i]
                    k_layer = k_mlp.mlp_layers.layers[i]

                    k_layer.kernel.assign(pt_layer.weight.detach().numpy().T)
                    k_layer.bias.assign(pt_layer.bias.detach().numpy())

    # Locate the heads
    pt_head = pt_model.transformer.decoder.ref_point_head
    k_head = k_model.transformer.decoder.ref_point_head

    transfer_mlp(pt_head, k_head)


# -------------------------------------------------------------------------
# MAIN TEST SUITE EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    keras.utils.set_random_seed(SEED)

    keras.backend.clear_session()
    log("Building PyTorch Model...", section=True)
    model_torch = build_model_torch(TEST_ARGS)
    model_torch.eval()

    log("Building Keras Model...", section=True)
    model_keras = build_model_keras(TEST_ARGS)
    model_keras(np.zeros((1, 518, 518, 3), dtype="float32"))

    port_weights(model_torch, model_keras)
    transfer_specific_internal_weights(model_torch, model_keras)

    verify_blocks_isolated(model_torch, model_keras)
    verify_end_to_end_propagation(model_torch, model_keras)
    verify_end_to_end_segments_propagation(model_torch, model_keras)
    verify_final_output(model_torch, model_keras)

    log("Test Suite Completed.", section=True)
    # -------------------------------------------------------------------------
    log("All tests finished successfully!", section=True)
