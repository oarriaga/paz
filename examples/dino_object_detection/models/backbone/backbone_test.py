import os
import sys
import re
import traceback

# --- Setup Environment ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Path Setup ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
sys.path.append(project_root)

import numpy as np
import torch
import torch.nn as nn
import keras
from keras import ops

# --- Imports ---
try:
    from examples.dino_object_detection.models.backbone.torch_backbone_for_testing import (
        build_backbone as build_backbone_torch,
        NestedTensor,
    )
    from examples.dino_object_detection.models.backbone.dinov2_backbone_wrapper import (
        build_backbone as build_backbone_keras,
    )
except ImportError:
    try:
        from torch_backbone_for_testing import (
            build_backbone as build_backbone_torch,
            NestedTensor,
        )
        from dinov2_backbone_wrapper import build_backbone as build_backbone_keras
    except ImportError:
        # Shim if needed
        from torch_backbone_for_testing import build_backbone as build_backbone_torch
        from dinov2_backbone_wrapper import build_backbone as build_backbone_keras

        class NestedTensor:
            def __init__(self, tensors, mask):
                self.tensors = tensors
                self.mask = mask

            def to(self, device):
                return self


# --- Configuration List ---
TEST_CONFIGS = [
    {
        "name": "dinov2_small",
        "encoder": "dinov2_small",
        "img_size": (518, 518),
        "patch_size": 14,
        "hidden_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "out_indices": [2, 5, 8, 11],
        "projector_scale": ["P3", "P4", "P5"],
        "num_register_tokens": 0,
    },
    # {
    #     "name": "dinov2_base",
    #     "encoder": "dinov2_base",
    #     "img_size": (518, 518),
    #     "patch_size": 14,
    #     "hidden_dim": 768,
    #     "depth": 12,
    #     "num_heads": 12,
    #     "out_indices": [2, 5, 8, 11],
    #     "projector_scale": ["P3", "P4", "P5"],
    #     "num_register_tokens": 0,
    # },
    # {
    #     "name": "dinov2_large",
    #     "encoder": "dinov2_large",
    #     "img_size": (518, 518),
    #     "patch_size": 14,
    #     "hidden_dim": 1024,
    #     "depth": 24,
    #     "num_heads": 16,
    #     "out_indices": [5, 11, 17, 23],
    #     "projector_scale": ["P3", "P4", "P5"],
    #     "num_register_tokens": 0,
    # },
    # {
    #     "name": "dinov2_small_registers",
    #     "encoder": "dinov2_small_registers",
    #     "img_size": (518, 518),
    #     "patch_size": 14,
    #     "hidden_dim": 384,
    #     "depth": 12,
    #     "num_heads": 6,
    #     "out_indices": [2, 5, 8, 11],
    #     "projector_scale": ["P3", "P4", "P5"],
    #     "num_register_tokens": 4,
    # },
    # {
    #     "name": "dinov2_base_registers",
    #     "encoder": "dinov2_base_registers",
    #     "img_size": (518, 518),
    #     "patch_size": 14,
    #     "hidden_dim": 768,
    #     "depth": 12,
    #     "num_heads": 12,
    #     "out_indices": [2, 5, 8, 11],
    #     "projector_scale": ["P3", "P4", "P5"],
    #     "num_register_tokens": 4,
    # },
    # {
    #     "name": "dinov2_large_registers",
    #     "encoder": "dinov2_large_registers",
    #     "img_size": (518, 518),
    #     "patch_size": 14,
    #     "hidden_dim": 1024,
    #     "depth": 24,
    #     "num_heads": 16,
    #     "out_indices": [5, 11, 17, 23],
    #     "projector_scale": ["P3", "P4", "P5"],
    #     "num_register_tokens": 4,
    # },
]

# Initialize global TEST_CONFIG with the first entry
TEST_CONFIG = TEST_CONFIGS[0].copy()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def log(msg, section=False, status=None):
    if section:
        print(f"\n{'='*80}\n{msg}\n{'='*80}")
    else:
        prefix = "[INFO]"
        if status == "PASS":
            prefix = "[PASS]"
        elif status == "FAIL":
            prefix = "[FAIL]"
        elif status == "WARN":
            prefix = "[WARN]"
        print(f"{prefix} {msg}")


def debug_stats(tensor, name):
    """Prints mean, std, min, max of a numpy array."""
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    tensor = np.array(tensor)
    print(
        f"      > {name:<20} | Shape: {str(tensor.shape):<15} | Mean: {tensor.mean():.4f} | Std: {tensor.std():.4f} | Range: [{tensor.min():.4f}, {tensor.max():.4f}]"
    )


def check_close(t_val, k_val, name, atol=1e-3, rtol=1e-3):
    """
    Compares PyTorch and Keras outputs.
    Returns (is_passed, max_diff)
    """
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

    # Shape Alignment: Keras(B,H,W,C) vs Torch(B,C,H,W)
    if k_np.ndim == 4 and t_np.ndim == 4:
        # Heuristic: if channels are last in Keras and first in Torch
        if k_np.shape[3] == t_np.shape[1] and k_np.shape[1] == t_np.shape[2]:
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
    msg = f"{name:<35} | Max Diff: {max_diff:.6f} | Mean: {mean_diff:.6f}"
    log(msg, status=status)
    return status == "PASS", max_diff


def set_weights(layer, weights_list):
    """Safe wrapper for Keras set_weights."""
    if "Identity" in layer.__class__.__name__:
        return False
    try:
        current_weights = layer.get_weights()
        if len(current_weights) != len(weights_list):
            log(
                f"Weight count mismatch for {layer.name}: Expect {len(current_weights)}, Got {len(weights_list)}",
                status="WARN",
            )
        layer.set_weights(weights_list)
        return True
    except Exception as e:
        log(f"Error setting weights for {layer.name}: {e}", status="FAIL")
        return False


def get_keras_blocks(model, depth):
    """Recursively finds transformer blocks in the Keras model."""
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
    """
    Extracts the functional backbone model and the wrapper layer
    from the Joiner model.
    """
    # 1. Unwrap Joiner -> Backbone Functional Model
    if hasattr(keras_model, "backbone"):
        # This is the 'backbone_joiner' functional model
        backbone_model = keras_model.backbone
    else:
        backbone_model = keras_model

    # 2. Get the Wrapper Layer
    try:
        wrapper = backbone_model.get_layer("dino_v2_backbone_wrapper")
    except ValueError:
        for layer in backbone_model.layers:
            if "DinoV2BackboneWrapper" in layer.__class__.__name__:
                wrapper = layer
                break
        else:
            raise ValueError("Could not find DinoV2BackboneWrapper in backbone model.")

    return backbone_model, wrapper


# -------------------------------------------------------------------------
# 1. Weight Porting
# -------------------------------------------------------------------------
def port_weights(torch_model, keras_model):
    log("Starting Robust Weight Porting (Debug Mode)...", section=True)

    full_sd = torch_model.state_dict()
    all_keys = list(full_sd.keys())

    # Check keys
    proj_keys = [k for k in all_keys if "projector" in k or "fpn" in k or "neck" in k]
    if not proj_keys:
        log("CRITICAL: No 'projector' keys found in PyTorch state dict!", status="FAIL")

    # ---------------------------------------------------------
    # PART 1: ENCODER
    # ---------------------------------------------------------
    t_backbone_wrapper = torch_model[0].encoder
    t_backbone = t_backbone_wrapper.encoder
    sd = t_backbone.state_dict()

    # --- FIX: Use helper to get models ---
    k_backbone_model, wrapper = get_backbone_and_wrapper(keras_model)
    k_model = getattr(wrapper, "feature_extractor", wrapper)

    # --- Positional Embeddings ---
    t_pos = sd["embeddings.position_embeddings"]
    k_pos_var = None
    for attr in ["positional_embedding", "pos_embed", "position_embeddings"]:
        if hasattr(k_model, attr):
            k_pos_var = getattr(k_model, attr)
            break
    if k_pos_var is not None:
        if t_pos.shape != k_pos_var.shape:
            # Interpolation logic
            cls_pos = t_pos[:, :1, :]
            patch_pos = t_pos[:, 1:, :]
            dim = t_pos.shape[-1]
            h_old = int(np.sqrt(patch_pos.shape[1]))
            h_new = int(np.sqrt(k_pos_var.shape[1] - 1))
            patch_pos = patch_pos.reshape(1, h_old, h_old, dim).permute(0, 3, 1, 2)
            patch_pos = torch.nn.functional.interpolate(
                patch_pos,
                size=(h_new, h_new),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
            t_pos = torch.cat([cls_pos, patch_pos], dim=1)
        k_pos_var.assign(t_pos.numpy())
        log("Ported Positional Embeddings", status="PASS")

    # --- CLS Token ---
    t_cls = sd["embeddings.cls_token"]
    for attr in ["class_token", "classification_token", "cls_token"]:
        if hasattr(k_model, attr):
            getattr(k_model, attr).assign(t_cls.numpy())
            log(f"Ported CLS Token ({attr})", status="PASS")
            break

    # --- Register Tokens ---
    if (
        hasattr(t_backbone.embeddings, "register_tokens")
        and t_backbone.embeddings.register_tokens is not None
    ):
        t_regs = t_backbone.embeddings.register_tokens.detach().numpy()
        if hasattr(k_model, "register_tokens") and k_model.register_tokens is not None:
            k_model.register_tokens.assign(t_regs)
            log(f"Ported Register Tokens {t_regs.shape}", status="PASS")
        else:
            log("Torch has registers but Keras model does not!", status="WARN")

    # --- Patch Embeddings ---
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

    # --- Blocks ---
    k_blocks = get_keras_blocks(keras_model, TEST_CONFIG["depth"])
    log(f"Found {len(k_blocks)} Keras Blocks")

    for i, k_blk in enumerate(k_blocks):
        prefix = f"encoder.layer.{i}."

        # Norm 1
        n1_w = sd[f"{prefix}norm1.weight"].numpy()
        n1_b = sd[f"{prefix}norm1.bias"].numpy()
        n1_layer = getattr(k_blk, "normalization1", getattr(k_blk, "norm1", None))
        if n1_layer:
            set_weights(n1_layer, [n1_w, n1_b])

        # Attention
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

        # LayerScale
        for ls_idx in [1, 2]:
            t_ls_key = f"{prefix}layer_scale{ls_idx}.lambda1"
            t_ls_val = sd.get(t_ls_key)
            if t_ls_val is not None:
                candidates = [
                    f"layer_scale_{ls_idx}",
                    f"ls{ls_idx}",
                    f"layerscale{ls_idx}",
                    f"layer_scale{ls_idx}",
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
                        # Attempt build
                        try:
                            c_dim = val_np.shape[0]
                            k_ls_layer.build((None, None, c_dim))
                            if hasattr(k_ls_layer, "gamma"):
                                target_var = k_ls_layer.gamma
                        except:
                            pass

                    if target_var is not None:
                        if target_var.shape != val_np.shape:
                            try:
                                val_np = val_np.reshape(target_var.shape)
                            except:
                                pass
                        target_var.assign(val_np)

        # Norm 2
        n2_w = sd[f"{prefix}norm2.weight"].numpy()
        n2_b = sd[f"{prefix}norm2.bias"].numpy()
        n2_layer = getattr(k_blk, "normalization2", getattr(k_blk, "norm2", None))
        if n2_layer:
            set_weights(n2_layer, [n2_w, n2_b])

        # MLP
        k_mlp = getattr(k_blk, "mlp", None)
        if k_mlp:
            fc1_w = sd.get(f"{prefix}mlp.fc1.weight")
            if fc1_w is not None:  # Standard
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
            else:  # SwiGLU
                win_w = sd[f"{prefix}mlp.weights_in.weight"].numpy().T
                win_b = sd[f"{prefix}mlp.weights_in.bias"].numpy()
                wout_w = sd[f"{prefix}mlp.weights_out.weight"].numpy().T
                wout_b = sd[f"{prefix}mlp.weights_out.bias"].numpy()
                k_win = getattr(
                    k_mlp,
                    "weights_in",
                    getattr(k_mlp, "fused_gate_and_value_projection", None),
                )
                k_wout = getattr(
                    k_mlp, "weights_out", getattr(k_mlp, "output_projection", None)
                )
                if k_win:
                    set_weights(k_win, [win_w, win_b])
                if k_wout:
                    set_weights(k_wout, [wout_w, wout_b])

    # Final Norm
    ln_w, ln_b = sd.get("layernorm.weight"), sd.get("layernorm.bias")
    if ln_w is not None:
        target = getattr(k_model, "normalization", getattr(k_model, "norm", None))
        if target:
            set_weights(target, [ln_w.numpy(), ln_b.numpy()])
        log("Ported Final Norm", status="PASS")

    # ---------------------------------------------------------
    # PART 2: PROJECTOR
    # ---------------------------------------------------------
    log("\n[INFO] Starting Projector (FPN) Weight Porting...", section=True)

    k_projector = None
    # --- FIX: Search in backbone layers, not Joiner layers ---
    for layer in k_backbone_model.layers:
        if "projector" in layer.name.lower() or "multiscale" in layer.name.lower():
            k_projector = layer
            break

    if k_projector is None:
        log("No Projector found in Keras model. Skipping.", status="WARN")
        return

    # ... [Rest of Projector copy logic remains the same] ...
    def copy_with_check(w_key, k_var, transpose=False):
        if w_key in full_sd:
            w_t = full_sd[w_key].numpy()
            if transpose and w_t.ndim == 4:
                w_t = w_t.transpose(2, 3, 1, 0)
            if k_var.shape != w_t.shape:
                log(
                    f"Shape Mismatch for {w_key}: Torch {w_t.shape} vs Keras {k_var.shape}",
                    status="WARN",
                )
                return
            k_var.assign(w_t)
        else:
            log(f"MISSING KEY: {w_key}", status="FAIL")

    def copy_single_layer(t_prefix, k_layer):
        if isinstance(k_layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)):
            if hasattr(k_layer, "kernel"):
                copy_with_check(f"{t_prefix}.weight", k_layer.kernel, transpose=True)
            else:
                copy_with_check(
                    f"{t_prefix}.weight", k_layer.weights[0], transpose=True
                )
            if hasattr(k_layer, "bias") and k_layer.bias is not None:
                copy_with_check(f"{t_prefix}.bias", k_layer.bias)
        elif (
            hasattr(k_layer, "gamma")
            or hasattr(k_layer, "beta")
            or hasattr(k_layer, "weight")
        ):
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

    # Detect prefix
    prefix_root = "projector"
    if not any(k.startswith("projector") for k in proj_keys):
        if len(proj_keys) > 0:
            prefix_root = proj_keys[0].split(".")[0]
            if "0" in prefix_root and len(prefix_root) < 3:
                prefix_root = (
                    proj_keys[0].split(".")[0] + "." + proj_keys[0].split(".")[1]
                )
            if "stages" in prefix_root:
                prefix_root = prefix_root.split("stages")[0].rstrip(".")
            log(f"Auto-detected Projector prefix: '{prefix_root}'", status="INFO")

    # Iterate Stages Sampling
    for i, stage_list in enumerate(k_projector.stages_sampling):
        for j, sampler_seq in enumerate(stage_list):
            t_base = f"{prefix_root}.stages_sampling.{i}.{j}"
            layer_idx = 0
            for layer in sampler_seq.layers:
                if isinstance(
                    layer, (keras.layers.Conv2DTranspose, keras.layers.Conv2D)
                ):
                    copy_single_layer(f"{t_base}.{layer_idx}", layer)
                    layer_idx += 1
                elif "LayerNorm" in layer.__class__.__name__:
                    copy_single_layer(f"{t_base}.{layer_idx}", layer)
                    layer_idx += 1
                elif hasattr(layer, "conv") and hasattr(layer, "bn"):
                    copy_convx(f"{t_base}.{layer_idx}", layer)
                    layer_idx += 1
                elif (
                    "Activation" in layer.__class__.__name__
                    or "GELU" in layer.__class__.__name__
                ):
                    layer_idx += 1

    # Iterate Stages (Fusion)
    for i, stage_seq in enumerate(k_projector.stages):
        t_base = f"{prefix_root}.stages.{i}"
        if len(stage_seq.layers) > 0:
            copy_c2f(f"{t_base}.0", stage_seq.layers[0])
        if len(stage_seq.layers) > 1:
            copy_single_layer(f"{t_base}.1", stage_seq.layers[1])

    log("Projector Weight Porting Complete", status="PASS")


# -------------------------------------------------------------------------
# 2. Detailed Verification
# -------------------------------------------------------------------------


def verify_layers(torch_model, keras_model):
    log("Starting Comprehensive Layer Verification...", section=True)

    B, C, H, W = 1, 3, TEST_CONFIG["img_size"][0], TEST_CONFIG["img_size"][1]
    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, (B, C, H, W)).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_keras_in = ops.convert_to_tensor(np.transpose(x_np, (0, 2, 3, 1)))

    t_backbone = torch_model[0].encoder.encoder

    # --- Unwrap backbone ---
    k_backbone_model, wrapper = get_backbone_and_wrapper(keras_model)
    k_model = getattr(wrapper, "feature_extractor", wrapper)

    # ---------------------------------------------------------------------
    # Step 1: Patch Embeddings
    # ---------------------------------------------------------------------
    log("[Check 1] Patch Embeddings / Projections")

    with torch.no_grad():
        t_patch_out = (
            t_backbone.embeddings.patch_embeddings(x_torch).detach().numpy()
        )  # (B, N, D)

    k_patch = None
    for l in k_model.layers:
        if "patch" in l.name.lower() or "conv" in l.name.lower():
            k_patch = l
            break

    if k_patch:
        k_patch_out = k_patch(x_keras_in)
        k_patch_out_np = ops.convert_to_numpy(k_patch_out)

        # Manually flatten Keras (B, H, W, C) -> (B, N, C) for comparison if needed
        if k_patch_out_np.ndim == 4:
            b, h, w, c = k_patch_out_np.shape
            k_patch_out_np = k_patch_out_np.reshape(b, h * w, c)

        check_close(t_patch_out, k_patch_out_np, "Patch Projection Output")
    else:
        log("Could not find Keras Patch Layer", status="WARN")

    # ---------------------------------------------------------------------
    # Step 2: Full Embeddings (Patch + CLS + Pos + Dropout)
    # ---------------------------------------------------------------------
    log("\n[Check 2] Full Embeddings (Input to Block 0)")

    torch_model.eval()

    with torch.no_grad():
        t_emb_out = t_backbone.embeddings(x_torch)  # (B, N+1, D)
        t_current_input = (
            t_emb_out.detach().numpy()
        )  # This is the Golden Input for Block 0

    # ---------------------------------------------------------------------
    # Step 3: Block-by-Block Verification (Isolated)
    # ---------------------------------------------------------------------
    k_blocks = get_keras_blocks(keras_model, TEST_CONFIG["depth"])
    t_layers = t_backbone.encoder.layer

    log(f"\n[Check 3] Verifying {len(k_blocks)} Blocks in Isolation...")
    log(
        "Strategy: Using PyTorch Output of Block N-1 as Input to Keras Block N.",
        status="INFO",
    )

    # This variable tracks the input to the current block (starts with Embedding out)
    current_block_input_np = t_current_input

    for i, (t_blk, k_blk) in enumerate(zip(t_layers, k_blocks)):
        print(f"\n--- Block {i} Debugging ---")

        # Prepare Inputs
        # Shape is (B, N, D) -> (1, 1370, 384) usually
        t_in_tensor = torch.from_numpy(current_block_input_np)
        k_in_tensor = ops.convert_to_tensor(current_block_input_np)

        # ------------------------------------------------------------
        # A. Run Full Block Comparison
        # ------------------------------------------------------------
        with torch.no_grad():
            t_out_full = t_blk(t_in_tensor)[0].detach().numpy()

        try:
            k_out_full = k_blk(k_in_tensor, training=False)
            k_out_full = ops.convert_to_numpy(k_out_full)
            passed, _ = check_close(t_out_full, k_out_full, f"Block {i} Full Output")
        except Exception as e:
            log(f"Block {i} Execution Error: {e}", status="FAIL")
            passed = False

        # ------------------------------------------------------------
        # B. Granular Debugging (Regardless of Pass/Fail)
        # ------------------------------------------------------------
        # 1. Norm 1
        with torch.no_grad():
            t_n1 = t_blk.norm1(t_in_tensor).detach().numpy()

        k_n1_layer = getattr(k_blk, "normalization1", getattr(k_blk, "norm1", None))
        if k_n1_layer:
            k_n1 = ops.convert_to_numpy(k_n1_layer(k_in_tensor))
            check_close(t_n1, k_n1, "  > Norm 1")

        # 2. Attention (Using Golden Norm1 Input)
        t_n1_tensor = torch.from_numpy(t_n1)
        k_n1_tensor = ops.convert_to_tensor(t_n1)

        with torch.no_grad():
            t_attn = t_blk.attention(t_n1_tensor)[0].detach().numpy()

        k_attn_layer = getattr(k_blk, "attention", None)
        if k_attn_layer:
            try:
                k_attn = k_attn_layer(k_n1_tensor, training=False)
                if isinstance(k_attn, (list, tuple)):
                    k_attn = k_attn[0]
                check_close(t_attn, k_attn, "  > Attention")
            except Exception as e:
                log(f"  > Attention Error: {e}", status="FAIL")

        # 3. Layer Scale 1
        t_attn_tensor = torch.from_numpy(t_attn)
        k_attn_tensor = ops.convert_to_tensor(t_attn)

        with torch.no_grad():
            t_ls1 = t_blk.layer_scale1(t_attn_tensor).detach().numpy()

        k_ls1_layer = getattr(k_blk, "layer_scale_1", getattr(k_blk, "ls1", None))
        if k_ls1_layer:
            k_ls1 = ops.convert_to_numpy(k_ls1_layer(k_attn_tensor))
            check_close(t_ls1, k_ls1, "  > LayerScale 1")

        # 4. Residual 1 Calculation (Manual Check)
        t_resid1 = current_block_input_np + t_ls1

        # 5. Norm 2 (Input is Residual 1)
        t_r1_tensor = torch.from_numpy(t_resid1)
        k_r1_tensor = ops.convert_to_tensor(t_resid1)

        with torch.no_grad():
            t_n2 = t_blk.norm2(t_r1_tensor).detach().numpy()

        k_n2_layer = getattr(k_blk, "normalization2", getattr(k_blk, "norm2", None))
        if k_n2_layer:
            k_n2 = ops.convert_to_numpy(k_n2_layer(k_r1_tensor))
            check_close(t_n2, k_n2, "  > Norm 2")

        # 6. MLP
        t_n2_tensor = torch.from_numpy(t_n2)
        k_n2_tensor = ops.convert_to_tensor(t_n2)

        with torch.no_grad():
            t_mlp = t_blk.mlp(t_n2_tensor).detach().numpy()

        k_mlp_layer = getattr(k_blk, "mlp", None)
        if k_mlp_layer:
            k_mlp = ops.convert_to_numpy(k_mlp_layer(k_n2_tensor))
            check_close(t_mlp, k_mlp, "  > MLP")

        # 7. Layer Scale 2
        t_mlp_tensor = torch.from_numpy(t_mlp)
        k_mlp_tensor = ops.convert_to_tensor(t_mlp)

        with torch.no_grad():
            t_ls2 = t_blk.layer_scale2(t_mlp_tensor).detach().numpy()

        k_ls2_layer = getattr(k_blk, "layer_scale_2", getattr(k_blk, "ls2", None))
        if k_ls2_layer:
            k_ls2 = ops.convert_to_numpy(k_ls2_layer(k_mlp_tensor))
            check_close(t_ls2, k_ls2, "  > LayerScale 2")

        # ------------------------------------------------------------
        # C. Advance Input
        # ------------------------------------------------------------
        current_block_input_np = t_out_full

    # ---------------------------------------------------------------------
    # Step 4: Final Normalization
    # ---------------------------------------------------------------------
    log("\n[Check 4] Final Normalization")
    t_final_in = torch.from_numpy(current_block_input_np)
    k_final_in = ops.convert_to_tensor(current_block_input_np)

    with torch.no_grad():
        t_final = t_backbone.layernorm(t_final_in).detach().numpy()

    k_norm = getattr(k_model, "normalization", getattr(k_model, "norm", None))
    if k_norm:
        k_final = ops.convert_to_numpy(k_norm(k_final_in))
        check_close(t_final, k_final, "Final Norm Output")
    else:
        log("Final Norm layer not found", status="WARN")


# -------------------------------------------------------------------------
# 3. Block-by-Block Verification (Outputs Only)
# -------------------------------------------------------------------------
def verify_blocks_isolated(torch_model, keras_model):
    log("Starting Block-by-Block Verification (Outputs Only)...", section=True)

    B, C, H, W = 1, 3, TEST_CONFIG["img_size"][0], TEST_CONFIG["img_size"][1]
    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, (B, C, H, W)).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    t_backbone = torch_model[0].encoder.encoder

    # --- FIX: Unwrap backbone ---
    k_backbone_model, wrapper = get_backbone_and_wrapper(keras_model)
    k_model = getattr(wrapper, "feature_extractor", wrapper)

    # 3. Get Initial Embeddings (The starting point)
    log("[Init] Computing Embeddings to serve as input for Block 0...")
    with torch.no_grad():
        t_current_input = t_backbone.embeddings(x_torch).detach().numpy()

    # 4. Get Blocks
    k_blocks = get_keras_blocks(keras_model, TEST_CONFIG["depth"])
    t_layers = t_backbone.encoder.layer

    log(f"Found {len(k_blocks)} Keras Blocks vs {len(t_layers)} Torch Layers.")

    # 5. Iterate Blocks
    for i, (t_blk, k_blk) in enumerate(zip(t_layers, k_blocks)):
        # Prepare inputs (Sequence: B, N, C)
        # We use the PyTorch output from previous step as input for BOTH to isolate errors
        t_in_tensor = torch.from_numpy(t_current_input)
        k_in_tensor = ops.convert_to_tensor(t_current_input)

        # Run Torch
        with torch.no_grad():
            # Torch blocks return tuple (hidden_states, attentions)
            t_out = t_blk(t_in_tensor)[0].detach().numpy()

        # Run Keras
        # Important: training=False to disable Stochastic Depth/Dropout
        k_out = k_blk(k_in_tensor, training=False)
        k_out = ops.convert_to_numpy(k_out)

        # Compare
        passed, max_diff = check_close(t_out, k_out, f"Block {i} Output")

        if not passed:
            log(
                f"   >>> Block {i} FAILED. Stopping chain might be advised if diff is huge.",
                status="WARN",
            )
            # Optional: detailed stats if it fails
            debug_stats(t_out, f"Torch Block {i}")
            debug_stats(k_out, f"Keras Block {i}")

        # Update input for next block (Teacher Forcing)
        # We use t_out (Golden Standard) so Block i+1 gets clean input
        t_current_input = t_out

    log("Block-by-Block Verification Complete.", section=True)


# -------------------------------------------------------------------------
# 4. End-to-End Propagation Verification (Drift Check)
# -------------------------------------------------------------------------
def verify_end_to_end_propagation(torch_model, keras_model):
    log("Starting End-to-End Error Propagation Check...", section=True)
    log("NOTE: This test does NOT reset inputs at each block.", status="INFO")
    log("It measures how error accumulates from Block 0 -> Block N.", status="INFO")

    # 1. Setup deterministic input
    B_val, C_val, H_val, W_val = (
        1,
        3,
        TEST_CONFIG["img_size"][0],
        TEST_CONFIG["img_size"][1],
    )

    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, (B_val, C_val, H_val, W_val)).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    t_backbone = torch_model[0].encoder.encoder

    # --- FIX: Unwrap backbone ---
    k_backbone_model, wrapper = get_backbone_and_wrapper(keras_model)
    k_model = getattr(wrapper, "feature_extractor", wrapper)

    # --- STEP 1: EMBEDDINGS (The Root of the Chain) ---
    log("\n[Stage 1] Embeddings (Patch + Pos + CLS + Registers)")

    # Torch Forward
    with torch.no_grad():
        t_state = t_backbone.embeddings(x_torch)  # (B, N, D)
        t_out = t_state.detach().numpy()

    # Keras Forward (Manual Reconstruction)
    try:
        # Check if model has a dedicated embeddings method (unlikely in this impl, but safe)
        if hasattr(k_model, "embeddings") and callable(k_model.embeddings):
            k_in = ops.convert_to_tensor(np.transpose(x_np, (0, 2, 3, 1)))
            k_state = k_model.embeddings(k_in)
        else:
            k_in = ops.convert_to_tensor(np.transpose(x_np, (0, 2, 3, 1)))

            # A. Patch Embed
            k_patch_layer = None
            for l in k_model.layers:
                if "patch" in l.name.lower() or "conv" in l.name.lower():
                    k_patch_layer = l
                    break

            if k_patch_layer is None:
                raise ValueError("Could not find Keras Patch Embedding layer")

            x = k_patch_layer(k_in)

            # Robust Shape Extraction
            input_shape = x.shape
            if len(input_shape) == 4:
                b, h, w, c = input_shape
                # Flatten (B, H, W, C) -> (B, N, C)
                x = ops.reshape(x, (b, h * w, c))
            elif len(input_shape) == 3:
                b, n, c = input_shape
            else:
                b = B_val
                c = TEST_CONFIG["hidden_dim"]

            # B. CLS Token
            cls_token = None
            for name in ["class_token", "classification_token", "cls_token"]:
                if hasattr(k_model, name):
                    cls_token = getattr(k_model, name)
                    break

            if cls_token is not None:
                cls_token = ops.cast(cls_token, x.dtype)
                cls_broadcast = ops.broadcast_to(cls_token, (b, 1, c))
                x = ops.concatenate([cls_broadcast, x], axis=1)

            # C. Pos Embed
            pos_embed = None
            for name in ["positional_embedding", "pos_embed", "position_embeddings"]:
                if hasattr(k_model, name):
                    pos_embed = getattr(k_model, name)
                    break

            if pos_embed is not None:
                # Handle interpolation if sizes differ
                if pos_embed.shape[1] != x.shape[1]:
                    if hasattr(k_model, "interpolate_positional_encoding"):
                        x = x + k_model.interpolate_positional_encoding(x, W_val, H_val)
                    else:
                        x = x + pos_embed
                else:
                    x = x + pos_embed

            # D. Register Tokens (CRITICAL FIX)
            register_tokens = None
            if hasattr(k_model, "register_tokens"):
                register_tokens = k_model.register_tokens

            # Check if it's not None (Keras weights can be None if conditional)
            if register_tokens is not None:
                register_tokens = ops.cast(register_tokens, x.dtype)
                # Keras DINOv2 usually inserts registers AFTER CLS (index 0) and BEFORE Patches (index 1:)
                # x shape here is [CLS, Patch1, Patch2...]

                reg_broadcast = ops.broadcast_to(
                    register_tokens, (b, register_tokens.shape[1], c)
                )

                # Split CLS and Patches
                cls_token_t = x[:, :1]
                patch_tokens_t = x[:, 1:]

                # Concatenate: [CLS, Registers, Patches]
                x = ops.concatenate(
                    [cls_token_t, reg_broadcast, patch_tokens_t], axis=1
                )

            k_state = x

        k_out = ops.convert_to_numpy(k_state)

        # Verify Shape Alignment
        if t_out.shape != k_out.shape:
            log(
                f"SHAPE MISMATCH: Torch {t_out.shape} vs Keras {k_out.shape}",
                status="FAIL",
            )
            # If shapes don't match, we cannot proceed with element-wise subtraction
            return

        passed, prev_diff = check_close(t_out, k_out, "Embeddings Output")
        if not passed:
            log(
                "CRITICAL: Embeddings already diverge. Propagation test will be noisy.",
                status="WARN",
            )

    except Exception as e:
        log(f"Could not isolate Embedding stage: {e}", status="FAIL")
        traceback.print_exc()
        return

    # --- STEP 2: BLOCK PROPAGATION ---
    k_blocks = get_keras_blocks(keras_model, TEST_CONFIG["depth"])
    t_layers = t_backbone.encoder.layer

    print(
        f"\n{'Block':<10} | {'Max Diff':<12} | {'Mean Diff':<12} | {'Drift Ratio':<12} | {'Status'}"
    )
    print("-" * 65)

    if prev_diff == 0:
        prev_diff = 1e-9

    for i, (t_blk, k_blk) in enumerate(zip(t_layers, k_blocks)):

        # 1. Torch Forward
        with torch.no_grad():
            t_state = t_blk(t_state)[0]
            t_out_np = t_state.detach().numpy()

        # 2. Keras Forward
        try:
            k_state = k_blk(k_state, training=False)
            k_out_np = ops.convert_to_numpy(k_state)
        except Exception as e:
            print(f"Block {i:<4} | EXECUTION FAILED: {e}")
            return

        # 3. Compare
        diff = np.abs(t_out_np.flatten() - k_out_np.flatten())
        curr_max_diff = np.max(diff)
        curr_mean_diff = np.mean(diff)

        # 4. Calculate Drift
        drift_ratio = curr_max_diff / (prev_diff + 1e-12)
        drift_str = f"{drift_ratio:.2f}x"

        status = "OK"
        if curr_max_diff > 1e-3:
            status = "HIGH"
        if curr_max_diff > 1e-1:
            status = "EXPLODED"

        print(
            f"Block {i:<4} | {curr_max_diff:.6f}     | {curr_mean_diff:.6f}     | {drift_str:<12} | {status}"
        )

        prev_diff = curr_max_diff

    # --- STEP 3: FINAL NORM ---
    log("\n[Stage 3] Final Normalization")

    with torch.no_grad():
        t_final = t_backbone.layernorm(t_state).detach().numpy()

    k_norm = getattr(k_model, "normalization", getattr(k_model, "norm", None))
    if k_norm:
        k_final = k_norm(k_state)
        k_final_np = ops.convert_to_numpy(k_final)

        diff = np.abs(t_final.flatten() - k_final_np.flatten())
        log(f"Final Output | Max Diff: {np.max(diff):.6f} | Mean: {np.mean(diff):.6f}")
    else:
        log("Final Norm layer not found.", status="WARN")


# -------------------------------------------------------------------------
# 5. Full Model "Black Box" Verification
# -------------------------------------------------------------------------
def verify_full_model_inference(torch_model, keras_model):
    log("Starting Full Model Black Box Verification (Joiner Mode)...", section=True)
    log("Checking Features AND Position Embeddings.", status="INFO")

    # 1. Setup Input
    B, C, H, W = 1, 3, TEST_CONFIG["img_size"][0], TEST_CONFIG["img_size"][1]
    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, (B, C, H, W)).astype(np.float32)

    # Torch: (B, C, H, W)
    x_torch = torch.from_numpy(x_np)
    # Keras: (B, H, W, C)
    x_keras = ops.convert_to_tensor(np.transpose(x_np, (0, 2, 3, 1)))

    # 2. Run Inference
    log("Running PyTorch Inference...")
    mask = torch.zeros((B, H, W), dtype=torch.bool)
    x_nested = NestedTensor(x_torch, mask)

    with torch.no_grad():
        # Torch returns (features: List[NestedTensor], pos: List[Tensor])
        t_out = torch_model(x_nested)
        t_features, t_pos_embeds = t_out

    log("Running Keras Inference...")
    # Keras returns (features: List[Tensor], pos: List[Tensor])
    k_out = keras_model(x_keras, training=False)
    k_features, k_pos_embeds = k_out

    # -------------------------------------------------------------------------
    # PART A: Verify Feature Maps
    # -------------------------------------------------------------------------
    log(f"\n[Verifying Feature Maps] Count: {len(t_features)} vs {len(k_features)}")

    for i, (t_feat_nested, k_feat) in enumerate(zip(t_features, k_features)):
        # Unwrap Torch NestedTensor -> Tensor (B, C, H, W)
        t_f_np = t_feat_nested.tensors.detach().cpu().numpy()
        k_f_np = ops.convert_to_numpy(k_feat)

        # Transpose Keras (B, H, W, C) -> (B, C, H, W) for comparison
        if k_f_np.ndim == 4 and k_f_np.shape[-1] == t_f_np.shape[1]:
            k_f_np = np.transpose(k_f_np, (0, 3, 1, 2))

        diff = np.abs(t_f_np - k_f_np)
        max_diff = np.max(diff)
        status = "PASS" if max_diff < 1e-3 else "FAIL"

        log(
            f"Feature Map {i} | Shape: {t_f_np.shape} | Max Diff: {max_diff:.6f}",
            status=status,
        )

    # -------------------------------------------------------------------------
    # PART B: Verify Position Embeddings
    # -------------------------------------------------------------------------
    log(
        f"\n[Verifying Position Embeds] Count: {len(t_pos_embeds)} vs {len(k_pos_embeds)}"
    )

    for i, (t_pos, k_pos) in enumerate(zip(t_pos_embeds, k_pos_embeds)):
        # Torch Pos: (B, C, H, W)
        t_p_np = t_pos.detach().cpu().numpy()
        k_p_np = ops.convert_to_numpy(k_pos)

        # Transpose Keras Pos (B, H, W, C) -> (B, C, H, W)
        # We implemented Keras Joiner to output channels-last to be Keras-idiomatic,
        # but PyTorch Joiner outputs channels-first.
        if k_p_np.ndim == 4 and k_p_np.shape[-1] == t_p_np.shape[1]:
            k_p_np = np.transpose(k_p_np, (0, 3, 1, 2))

        diff = np.abs(t_p_np - k_p_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        status = (
            "PASS" if max_diff < 1e-4 else "FAIL"
        )  # Pos embeds usually very precise
        log(
            f"Pos Embed {i}   | Shape: {t_p_np.shape} | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f}",
            status=status,
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    for i, config_dict in enumerate(TEST_CONFIGS):
        # Update the global configuration object so helper functions (which rely on TEST_CONFIG) work correctly
        TEST_CONFIG.update(config_dict)

        print(f"\n{'#'*80}")
        print(f"Running Test Suite {i+1}/{len(TEST_CONFIGS)}: {TEST_CONFIG['name']}")
        print(f"{'#'*80}\n")

        # Reset Keras session to avoid clutter/OOM
        keras.backend.clear_session()

        try:
            log(f"Building PyTorch Model ({TEST_CONFIG['encoder']})...", section=True)
            model_torch = build_backbone_torch(
                encoder=TEST_CONFIG["encoder"],
                vit_encoder_num_layers=TEST_CONFIG["depth"],
                pretrained_encoder=None,
                window_block_indexes=[],
                drop_path=0.0,
                out_channels=256,
                out_feature_indexes=TEST_CONFIG["out_indices"],
                projector_scale=TEST_CONFIG["projector_scale"],
                use_cls_token=True,
                hidden_dim=TEST_CONFIG["hidden_dim"],
                position_embedding="sine",
                freeze_encoder=False,
                layer_norm=True,
                target_shape=TEST_CONFIG["img_size"],
                rms_norm=False,
                backbone_lora=False,
                force_no_pretrain=False,
                gradient_checkpointing=False,
                load_dinov2_weights=True,
                patch_size=TEST_CONFIG["patch_size"],
                num_windows=1,
                positional_encoding_size=37,
                # Note: Torch impl likely reads config or has a default,
                # but if your torch builder supports it, pass it.
                # Assuming your provided torch code handles it via config/name.
            )
            model_torch.eval()

            log(f"Building Keras Model ({TEST_CONFIG['encoder']})...", section=True)
            model_keras = build_backbone_keras(
                encoder=TEST_CONFIG["encoder"],
                vit_encoder_num_layers=TEST_CONFIG["depth"],
                pretrained_encoder=None,
                window_block_indexes=[],
                drop_path=0.0,
                out_channels=256,
                out_feature_indexes=TEST_CONFIG["out_indices"],
                projector_scale=TEST_CONFIG["projector_scale"],
                use_cls_token=True,
                hidden_dim=TEST_CONFIG["hidden_dim"],
                position_embedding="sine",
                freeze_encoder=False,
                layer_norm=True,
                target_shape=TEST_CONFIG["img_size"],
                rms_norm=False,
                backbone_lora=False,
                force_no_pretrain=True,
                gradient_checkpointing=False,
                load_dinov2_weights=False,
                patch_size=TEST_CONFIG["patch_size"],
                num_windows=1,
                positional_encoding_size=37,
                init_values=1e-5,
                # --- FIX: Pass the register tokens count ---
                num_register_tokens=TEST_CONFIG.get("num_register_tokens", 0),
            )

            # Init with dummy forward pass
            # dummy = np.zeros((1, *TEST_CONFIG["img_size"], 3), dtype="float32")
            dummy = np.random.uniform(-1, 1, (1, *TEST_CONFIG["img_size"], 3)).astype(np.float32)
            model_keras(dummy)

            # Run Tests
            port_weights(model_torch, model_keras)

            verify_layers(model_torch, model_keras)
            verify_blocks_isolated(model_torch, model_keras)
            verify_end_to_end_propagation(model_torch, model_keras)
            verify_full_model_inference(model_torch, model_keras)

            log(f"\n>>> Test Suite for {TEST_CONFIG['name']} FINISHED.", section=True)

        except Exception:
            print(f"\n!!! CRITICAL FAILURE in configuration {TEST_CONFIG['name']} !!!")
            traceback.print_exc()
            print("Continuing to next configuration...\n")
