"""Convert the FLOWER LIBERO checkpoint DiT into a paz weight file.

Dev-only script: reads the flat safetensors state dict of the upstream
FLOWERVLA LightningModule and fills the paz flow DiT built by
``model.build``. Every non-VLM checkpoint tensor is either mapped,
verified, or listed in SKIPPED_PREFIXES with a reason; anything else
aborts the conversion. ``vlm.*`` tensors belong to the florence2
conversion and are ignored here.

Notes discovered from the reference implementation:
- ``adaln.<space>.modCX`` emits 9 chunks but FlowBlock consumes only the
  first 6 (transformers.py:392), so only 6 * hidden rows are loaded.
- ``dit.N.self_attn.cos/sin`` RoPE buffers are loaded, not recomputed:
  blocks 0-11 hold wavelength-1000 tables (bf16-rounded, inherited from
  pretraining) while blocks 12-17 hold the configured wavelength-32
  tables. The script reports the match per block.
- ``action_space_embedder.*`` and the non-``eef_delta`` action heads are
  dead at inference (never referenced by ``dit_forward``).
"""
import argparse
from pathlib import Path

import numpy as np

from paz.models.foundation.flower.model import build

NUM_LAYERS = 18
HIDDEN_DIM = 1024
ACTION_SPACE = "eef_delta"

SKIPPED_PREFIXES = {
    "action_space_embedder.":
        "unused at inference: dit_forward never calls it",
    "action_encoders.joint_single.": "inactive action space",
    "action_encoders.bimanual_nav.": "inactive action space",
    "action_decoders.joint_single.": "inactive action space",
    "action_decoders.bimanual_nav.": "inactive action space",
    "adaln.joint_single.": "inactive action space",
    "adaln.bimanual_nav.": "inactive action space",
}


def convert(checkpoint_path, output_dir):
    from safetensors import safe_open
    checkpoint = safe_open(str(checkpoint_path), framework="numpy")
    model = build()
    used = transfer(checkpoint, model)
    rope_report = report_rope_tables(checkpoint)
    skipped = audit_keys(checkpoint, used)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_dir / "flower_dit.weights.h5"))
    report = build_report(used, skipped, rope_report)
    (output_dir / "flower_dit_keymap.txt").write_text(report)
    return model


def transfer(checkpoint, model):
    used = {}

    def tensor(key, target):
        used[key] = target
        return checkpoint.get_tensor(key)

    def set_dense(layer_name, key):
        layer = model.get_layer(layer_name)
        weights = [tensor(f"{key}.weight", layer_name).T]
        if layer.use_bias:
            weights.append(tensor(f"{key}.bias", layer_name))
        layer.set_weights(weights)

    def set_norm(layer_name, key):
        scale = tensor(f"{key}.weight", layer_name)
        model.get_layer(layer_name).set_weights([scale])

    set_dense("flow_time_embedder_dense_1", "t_embedder.mlp.0")
    set_dense("flow_time_embedder_dense_2", "t_embedder.mlp.2")
    set_dense("frequency_embedder_dense_1", "frequency_embedder.mlp.0")
    set_dense("frequency_embedder_dense_2", "frequency_embedder.mlp.2")
    set_norm("context_norm", "cond_norm")
    set_dense("context_projection", "cond_linear")
    set_dense("action_encoder_fc1", f"action_encoders.{ACTION_SPACE}.fc1")
    set_dense("action_encoder_fc2", f"action_encoders.{ACTION_SPACE}.fc2")
    set_dense("action_decoder", f"action_decoders.{ACTION_SPACE}")
    transfer_shared_adaln(tensor, model)
    for block in range(NUM_LAYERS):
        transfer_block(tensor, set_dense, set_norm, model, block)
    return used


def transfer_shared_adaln(tensor, model):
    key = f"adaln.{ACTION_SPACE}.modCX.1.weight"
    note = ("shared_adaln (first 6 of 9 chunks; chunks 6-8 are dead "
            "in FlowBlock.forward)")
    weight = tensor(key, note)
    if weight.shape[0] != 9 * HIDDEN_DIM:
        raise ValueError(f"unexpected shared adaln shape {weight.shape}")
    num_used_rows = 6 * HIDDEN_DIM
    model.get_layer("shared_adaln").set_weights([weight[:num_used_rows].T])


def transfer_block(tensor, set_dense, set_norm, model, block):
    source = f"dit.{block}"
    target = f"block_{block}"
    set_dense(f"{target}_adaln_dense_1", f"{source}.adaLN_modulation.1")
    set_dense(f"{target}_adaln_dense_2", f"{source}.adaLN_modulation.2")
    set_norm(f"{target}_norm_1", f"{source}.norm1")
    set_norm(f"{target}_norm_2", f"{source}.norm2")
    set_norm(f"{target}_norm_3", f"{source}.norm3")
    transfer_self_attention(tensor, set_norm, model, source, target)
    cross = f"{target}_cross_attention"
    set_dense(f"{cross}_query", f"{source}.cross_attn.q_proj")
    set_dense(f"{cross}_key", f"{source}.cross_attn.k_proj")
    set_dense(f"{cross}_value", f"{source}.cross_attn.v_proj")
    set_dense(f"{cross}_output", f"{source}.cross_attn.proj")
    set_norm(f"{cross}_query_norm", f"{source}.cross_attn.q_norm")
    set_norm(f"{cross}_key_norm", f"{source}.cross_attn.k_norm")
    set_dense(f"{target}_mlp_gate", f"{source}.mlp.fc1")
    set_dense(f"{target}_mlp_up", f"{source}.mlp.fc2")
    set_dense(f"{target}_mlp_down", f"{source}.mlp.proj")


def transfer_self_attention(tensor, set_norm, model, source, target):
    name = f"{target}_self_attention"
    fused = tensor(f"{source}.self_attn.qkv.weight", f"{name}_qkv")
    model.get_layer(f"{name}_qkv").set_weights([fused.T])
    set_norm(f"{name}_query_norm", f"{source}.self_attn.q_norm")
    set_norm(f"{name}_key_norm", f"{source}.self_attn.k_norm")
    output = tensor(f"{source}.self_attn.proj.weight", f"{name}_output")
    model.get_layer(f"{name}_output").set_weights([output.T])
    cosine = tensor(f"{source}.self_attn.cos", f"{name}_rotary")
    sine = tensor(f"{source}.self_attn.sin", f"{name}_rotary")
    model.get_layer(f"{name}_rotary").set_weights([cosine, sine])


def report_rope_tables(checkpoint):
    lines = []
    for block in range(NUM_LAYERS):
        cosine = checkpoint.get_tensor(f"dit.{block}.self_attn.cos")
        sine = checkpoint.get_tensor(f"dit.{block}.self_attn.sin")
        errors = {}
        for wavelength in (32.0, 1000.0):
            reference = build_rope_tables(wavelength, cosine.shape)
            cosine_error = np.abs(cosine - reference[0]).max()
            sine_error = np.abs(sine - reference[1]).max()
            errors[wavelength] = max(cosine_error, sine_error)
        wavelength = min(errors, key=errors.get)
        line = (f"dit.{block}.self_attn.cos/sin: closest wavelength "
                f"{wavelength:g} (max err {errors[wavelength]:.2e})")
        lines.append(line)
        if errors[wavelength] > 2e-3:
            raise ValueError(f"unrecognized RoPE table in block {block}")
    return lines


def build_rope_tables(wavelength, shape):
    num_positions, half = shape
    head_dim = 2 * half
    indices = np.arange(0, head_dim, 2, dtype=np.float32)
    frequencies = np.float32(wavelength) ** (-indices / np.float32(head_dim))
    positions = np.arange(num_positions, dtype=np.float32)
    angles = positions[:, None] * frequencies[None]
    return np.cos(angles), np.sin(angles)


def audit_keys(checkpoint, used):
    skipped = {}
    unmapped = []
    for key in checkpoint.keys():
        if key.startswith("vlm.") or key in used:
            continue
        reason = skip_reason(key)
        if reason is None:
            unmapped.append(key)
        else:
            skipped[key] = reason
    if unmapped:
        raise ValueError(f"unmapped checkpoint keys: {sorted(unmapped)}")
    return skipped


def skip_reason(key):
    for prefix, reason in SKIPPED_PREFIXES.items():
        if key.startswith(prefix):
            return reason
    return None


def build_report(used, skipped, rope_report):
    lines = ["# FLOWER DiT keymap (torch key -> paz layer)"]
    for key in sorted(used):
        lines.append(f"{key} -> {used[key]}")
    lines.append("")
    lines.append("# Skipped checkpoint keys")
    for key in sorted(skipped):
        lines.append(f"{key}: {skipped[key]}")
    lines.append("")
    lines.append("# RoPE buffer provenance (loaded verbatim)")
    lines.extend(rope_report)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--checkpoint", required=True, help="path to model.safetensors")
    add("--output_dir", required=True, help="directory for weights + report")
    add("--tokenizer", default=None,
        help="Florence-2 tokenizer.json to copy into the artifact dir")
    args = parser.parse_args()
    convert(args.checkpoint, args.output_dir)
    if args.tokenizer is not None:
        import shutil
        shutil.copy(args.tokenizer, Path(args.output_dir) / "tokenizer.json")
    print(f"wrote {args.output_dir}/flower_dit.weights.h5")
