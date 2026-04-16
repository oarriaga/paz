import os
import sys

# Restart with CPU-only JAX to avoid GPU OOM during model graph construction.
# Must happen before any JAX/Keras initialization.
if os.environ.get("JAX_PLATFORMS") != "cpu":
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "KERAS_BACKEND": "jax"}
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

os.environ["KERAS_BACKEND"] = "jax"

import json
import argparse
from pathlib import Path

import numpy as np


E2B_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--google--gemma-4-E2B-it"
    / "snapshots"
    / "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
)

E4B_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--google--gemma-4-E4B-it"
    / "snapshots"
    / "83df0a889143b1dbfc61b591bbc639540fd9ce4c"
)

HF_MODEL_LOCATION = E2B_SNAPSHOT


def open_safetensors(model_id_or_path):
    """Open a safetensors file for streaming access (no full RAM load)."""
    from safetensors import safe_open
    sf_path = Path(model_id_or_path) / "model.safetensors"
    handle = safe_open(str(sf_path), framework="numpy", device="cpu")
    return handle


def extract_config_from_hf(hf_config_path):
    """Build TextBackboneArgs from HF config.json text_config section."""
    from examples.gemma4.model import TextBackboneArgs
    with open(str(hf_config_path)) as f:
        hf = json.load(f)
    tc = hf["text_config"]
    layer_types = tc["layer_types"]
    num_layers = len(layer_types)
    global_layer_indices = _global_layer_indices(layer_types)
    pattern = _infer_sliding_window_pattern(layer_types)
    return TextBackboneArgs(
        vocabulary_size=tc["vocab_size"],
        image_size=896,
        num_layers=num_layers,
        num_query_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        hidden_dim=tc["hidden_size"],
        intermediate_dim=tc["intermediate_size"],
        head_dim=tc["head_dim"],
        attention_logit_soft_cap=None,
        final_logit_soft_cap=tc.get("final_logit_softcapping"),
        use_sliding_window_attention=True,
        sliding_window_size=tc["sliding_window"],
        sliding_window_pattern=pattern,
        global_head_dim=tc.get("global_head_dim"),
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        global_rope_partial_rotary_factor=tc["rope_parameters"]
            .get("full_attention", {}).get("partial_rotary_factor", 1.0),
        use_bidirectional_attention=False,
        layer_norm_epsilon=tc["rms_norm_eps"],
        dropout=tc.get("attention_dropout", 0.0),
        dtype="bfloat16",
        hidden_size_per_layer_input=tc.get("hidden_size_per_layer_input"),
        num_kv_shared_layers=tc.get("num_kv_shared_layers", 0) or 0,
        global_layer_indices=global_layer_indices,
    )


def _global_layer_indices(layer_types):
    """Return explicit global indices only when pattern is irregular.

    When global layers follow a regular modular pattern (every N layers),
    return None so the sliding_window_pattern field is used instead.
    This avoids storing a large tuple for regular architectures.
    """
    indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    if not indices:
        return None
    # Check if indices follow a regular pattern (every N layers, first at N-1)
    n = len(layer_types)
    first = indices[0]
    pattern = first + 1
    expected = list(range(first, n, pattern))
    if indices == expected:
        return None  # regular — use sliding_window_pattern instead
    return tuple(indices)


def _infer_sliding_window_pattern(layer_types):
    """Infer sliding_window_pattern from the HF layer_types list."""
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            return i + 1
    return len(layer_types)


def build_paz_weight_map(sf_handle, config, paz_paths=None):
    """Yield (paz_path, numpy_array) pairs one at a time for streaming assign.

    paz_paths: optional set of paz weight paths the target model actually has.
    When provided, tensors whose paz_path is NOT in paz_paths are skipped
    entirely — the safetensors _get call is never made, avoiding large RAM
    spikes for tensors that belong to the other model half.

    Using a generator keeps only one tensor in RAM beyond the paz model.
    """
    from examples.gemma4.model import (is_global_attention_layer,
                                        build_feedforward_dim)
    prefix = "model.language_model"

    paz_path = "token_embedding/embeddings"
    if paz_paths is None or paz_path in paz_paths:
        yield (paz_path,
               _get(sf_handle, "{}.embed_tokens.weight".format(prefix)))

    if config.hidden_size_per_layer_input:
        paz_path = "per_layer_embeddings/embeddings"
        if paz_paths is None or paz_path in paz_paths:
            key = "{}.embed_tokens_per_layer.weight".format(prefix)
            arr = _get(sf_handle, key)
            expected_dim = config.num_layers * config.hidden_size_per_layer_input
            if arr.shape != (config.vocabulary_size, expected_dim):
                raise ValueError(
                    "embed_tokens_per_layer shape mismatch: "
                    "expected ({}, {}), got {}".format(
                        config.vocabulary_size, expected_dim, arr.shape))
            yield (paz_path, arr)

        # Per-layer model projection: HF (n, d) → paz (d, n) for btd,dn->btn
        paz_path = "per_layer_model_projection/kernel"
        if paz_paths is None or paz_path in paz_paths:
            yield (paz_path,
                   _get(sf_handle,
                        "{}.per_layer_model_projection.weight".format(prefix)).T)

        paz_path = "per_layer_projection_norm/scale"
        if paz_paths is None or paz_path in paz_paths:
            yield (paz_path,
                   _get(sf_handle,
                        "{}.per_layer_projection_norm.weight".format(prefix)))

    yield ("final_normalization/scale",
           _get(sf_handle, "{}.norm.weight".format(prefix)))

    for i in range(config.num_layers):
        hf_layer = "{}.layers.{}".format(prefix, i)
        paz = "decoder_block_{}".format(i)
        is_global = is_global_attention_layer(config, i)
        head_dim = (config.global_head_dim
                    if (is_global and config.global_head_dim)
                    else config.head_dim)
        d = config.hidden_dim
        n = config.num_query_heads
        k = config.num_key_value_heads
        f = build_feedforward_dim(config, i)
        for item in _layer_weight_pairs(sf_handle, hf_layer, paz,
                                         d, n, k, head_dim, f, config):
            yield item


def _layer_weight_pairs(sf_handle, hf_layer, paz, d, n, k, head_dim, f, config):
    """Yield (paz_path, array) pairs for one decoder layer."""
    norms = [
        ("input_layernorm.weight",
         "{}_pre_attention_norm/scale".format(paz)),
        ("post_attention_layernorm.weight",
         "{}_post_attention_norm/scale".format(paz)),
        ("pre_feedforward_layernorm.weight",
         "{}_pre_ffw_norm/scale".format(paz)),
        ("post_feedforward_layernorm.weight",
         "{}_post_ffw_norm/scale".format(paz)),
        ("self_attn.q_norm.weight",
         "{}_attention_query_norm/scale".format(paz)),
        ("self_attn.k_norm.weight",
         "{}_attention_key_norm/scale".format(paz)),
    ]
    for hf_key, paz_path in norms:
        yield (paz_path,
               _get(sf_handle, "{}.{}".format(hf_layer, hf_key)))

    # Q: HF (n*h, d) -> paz (n, d, h) for einsum btd,ndh->btnh
    q_raw = _get(sf_handle,
                 "{}.self_attn.q_proj.weight".format(hf_layer))
    yield ("{}_attention_query/kernel".format(paz),
           q_raw.reshape(n, head_dim, d).transpose(0, 2, 1))

    # K: HF (k*h, d) -> paz (k, d, h) for einsum btd,kdh->btkh
    k_raw = _get(sf_handle,
                 "{}.self_attn.k_proj.weight".format(hf_layer))
    yield ("{}_attention_key/kernel".format(paz),
           k_raw.reshape(k, head_dim, d).transpose(0, 2, 1))

    # V: HF (k*h, d) -> paz (k, d, h)
    v_raw = _get(sf_handle,
                 "{}.self_attn.v_proj.weight".format(hf_layer))
    yield ("{}_attention_value/kernel".format(paz),
           v_raw.reshape(k, head_dim, d).transpose(0, 2, 1))

    # O: HF (d, n*h) -> paz (n, h, d) for einsum btnh,nhd->btd
    o_raw = _get(sf_handle,
                 "{}.self_attn.o_proj.weight".format(hf_layer))
    yield ("{}_attention_attention_output/kernel".format(paz),
           o_raw.T.reshape(n, head_dim, d))

    # FFW gate/up: HF (f, d) -> paz (d, f)
    yield ("{}_ffw_gating/kernel".format(paz),
           _get(sf_handle,
                "{}.mlp.gate_proj.weight".format(hf_layer)).T)
    yield ("{}_ffw_gating_2/kernel".format(paz),
           _get(sf_handle,
                "{}.mlp.up_proj.weight".format(hf_layer)).T)

    # FFW down: HF (d, f) -> paz (f, d)
    yield ("{}_ffw_linear/kernel".format(paz),
           _get(sf_handle,
                "{}.mlp.down_proj.weight".format(hf_layer)).T)

    # Layer scalar: HF shape (1,) -> paz shape ()
    yield ("{}_layer_scalar/scale".format(paz),
           _get(sf_handle,
                "{}.layer_scalar".format(hf_layer)).squeeze())

    # Per-layer input weights (E4B/E2B)
    pl = config.hidden_size_per_layer_input
    if pl:
        g_raw = _get(
            sf_handle,
            "{}.per_layer_input_gate.weight".format(hf_layer))
        yield ("{}_per_layer_gate/kernel".format(paz),
               g_raw.T if g_raw.shape == (pl, d) else g_raw)

        p_raw = _get(
            sf_handle,
            "{}.per_layer_projection.weight".format(hf_layer))
        yield ("{}_per_layer_projection/kernel".format(paz),
               p_raw.T if p_raw.shape == (d, pl) else p_raw)

        yield ("{}_post_per_layer_norm/scale".format(paz),
               _get(sf_handle,
                    "{}.post_per_layer_input_norm.weight".format(hf_layer)))


def _get(sf_handle, key):
    """Fetch one tensor from safetensors handle in its native dtype.

    Keeping bfloat16 avoids a 2x float32 temporary buffer. The paz model
    weights are bfloat16, so there is no dtype mismatch during assign().
    """
    if key not in sf_handle.keys():
        raise KeyError("safetensors missing key: '{}'".format(key))
    return sf_handle.get_tensor(key)


def load_hf_weights_into_paz(paz_model, sf_handle, config):
    """Stream HF weights into paz model one tensor at a time.

    Weights in the HF map that are not present in paz_model are silently
    skipped — this allows loading each half of a split model independently
    (e.g. Gemma4PerLayerEmbStep vs Gemma4DecoderStep).
    """
    paz_weights = {w.path: w for w in paz_model.weights}
    assigned = set()
    for paz_path, value in build_paz_weight_map(
            sf_handle, config, paz_paths=set(paz_weights.keys())):
        if paz_path not in paz_weights:
            del value
            continue  # belongs to the other model half
        paz_weights[paz_path].assign(value)
        assigned.add(paz_path)
        del value
    unassigned = [p for p in paz_weights if p not in assigned]
    if unassigned:
        raise ValueError(
            "paz model weights not covered by HF map:\n  {}".format(
                "\n  ".join(unassigned)))
    return paz_model


def export_from_hf(model_id_or_path, output_dir, config=None):
    """Load HF model weights and export paz config.json + model.weights.h5."""
    from examples.gemma4.inference import Gemma4DecoderStep
    from examples.gemma4.reference import save_config

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_id_or_path)
    hf_config_path = (model_path / "config.json"
                      if model_path.is_dir()
                      else HF_MODEL_LOCATION / "config.json")

    if config is None:
        config = extract_config_from_hf(hf_config_path)

    print("Building paz decoder step model (CPU)...", flush=True)
    step = Gemma4DecoderStep(config)

    print("Opening HF safetensors (streaming)...", flush=True)
    sf_handle = open_safetensors(model_id_or_path)

    print("Mapping weights to paz format...", flush=True)
    load_hf_weights_into_paz(step, sf_handle, config)

    weights_path = output_dir / "model.weights.h5"
    print("Saving weights to {}...".format(weights_path), flush=True)
    step.save_weights(str(weights_path))

    config_path = output_dir / "config.json"
    save_config(config, config_path)
    print("Saved config to {}.".format(config_path), flush=True)

    return config, step


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    description = "Export HuggingFace Gemma 4 weights to paz format"
    parser = argparse.ArgumentParser(description=description)
    add = parser.add_argument
    add("model_id", help=(
        "HF model ID or local path to snapshot directory"))
    add("--output_dir", default=None)
    add("--config", default="gemma4_e2b_1b_it")
    args = parser.parse_args()

    from examples.gemma4.configuration import CONFIGS
    from examples.gemma4.model import TextBackboneArgs

    config_dict = CONFIGS.get(args.config)
    if config_dict is None:
        raise ValueError("Unknown config: '{}'".format(args.config))
    cfg = TextBackboneArgs(**config_dict)

    output = args.output_dir
    if output is None:
        models_dir = Path(__file__).resolve().with_name("gemma4_models")
        output = models_dir / args.config

    export_from_hf(args.model_id, output, config=cfg)
    print("Export complete. Output:", output)
