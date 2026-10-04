"""Convert the official Gemma4 12B text checkpoint into paz weight files.

Run this with `safetensors` and `torch` installed; the paz runtime needs
neither. It writes the artifacts `Gemma4(...)` loads: config.json,
tokenizer.json and backbone.weights.h5. The unified vision and audio
projections are text-irrelevant and are skipped.
"""
import argparse
import shutil
from pathlib import Path

import jax

from paz.models.foundation.gemma4.configuration import save_config
from paz.models.foundation.gemma4.configuration import to_backbone_args
from paz.models.foundation.gemma4.conversion import build_target_backbone

TEXT_PREFIX = "model.language_model."
NORM_NAMES = {
    "pre_attention_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "pre_ffw_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "query_norm": "self_attn.q_norm",
    "key_norm": "self_attn.k_norm",
}


def convert(checkpoint_dir, output_dir, model_name="gemma4_12b"):
    checkpoint_dir, output_dir = Path(checkpoint_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = to_backbone_args(model_name)
    save_config(config, output_dir / "config.json")
    tokenizer = "tokenizer.json"
    shutil.copy(checkpoint_dir / tokenizer, output_dir / tokenizer)
    # Keep the whole 24 GB transfer off the accelerator.
    with jax.default_device(jax.devices("cpu")[0]):
        backbone = build_target_backbone(config)
        save_backbone(checkpoint_dir, backbone, output_dir)
    return config


def save_backbone(checkpoint_dir, backbone, output_dir):
    from safetensors import safe_open
    path = str(Path(checkpoint_dir) / "model.safetensors")
    with safe_open(path, framework="pt") as source:
        assert_text_weights_match(source, backbone)
        transfer(source, backbone)
    backbone.save_weights(str(Path(output_dir) / "backbone.weights.h5"))


def assert_text_weights_match(source, backbone):
    names = [name for name in source.keys() if name.startswith(TEXT_PREFIX)]
    expected, found = len(backbone.weights), len(names)
    if expected != found:
        message = "expected {} text weights, checkpoint has {}"
        raise ValueError(message.format(expected, found))


def transfer(source, backbone):
    embeddings = read(source, "embed_tokens.weight")
    backbone.token_embedding.embeddings.assign(embeddings)
    backbone.final_normalization.scale.assign(read(source, "norm.weight"))
    for index, layer in enumerate(backbone.decoder_layers):
        transfer_layer(source, layer, "layers.{}.".format(index))


def transfer_layer(source, layer, prefix):
    for attribute, name in NORM_NAMES.items():
        scale = read(source, prefix + name + ".weight")
        getattr(layer, attribute).scale.assign(scale)
    layer.layer_scalar.scale.assign(read(source, prefix + "layer_scalar")[0])
    transfer_attention(source, layer, prefix + "self_attn.")
    transfer_feedforward(source, layer, prefix + "mlp.")


def transfer_attention(source, layer, prefix):
    num_kv_heads, head_dim = layer.kv_shape
    num_heads = layer.config.num_query_heads
    query = read(source, prefix + "q_proj.weight")
    layer.query_proj.kernel.assign(to_heads(query, num_heads, head_dim))
    key = read(source, prefix + "k_proj.weight")
    layer.key_proj.kernel.assign(to_heads(key, num_kv_heads, head_dim))
    if layer.value_proj is not None:
        value = read(source, prefix + "v_proj.weight")
        layer.value_proj.kernel.assign(to_heads(value, num_kv_heads, head_dim))
    output = read(source, prefix + "o_proj.weight")
    layer.output_proj.kernel.assign(to_output(output, num_heads, head_dim))


def transfer_feedforward(source, layer, prefix):
    layer.ffw_gating.kernel.assign(read(source, prefix + "gate_proj.weight").T)
    layer.ffw_gating_2.kernel.assign(read(source, prefix + "up_proj.weight").T)
    layer.ffw_linear.kernel.assign(read(source, prefix + "down_proj.weight").T)


def read(source, name):
    return source.get_tensor(TEXT_PREFIX + name).float().numpy()


def to_heads(kernel, num_heads, head_dim):
    return kernel.reshape(num_heads, head_dim, -1).transpose(0, 2, 1)


def to_output(kernel, num_heads, head_dim):
    return kernel.reshape(-1, num_heads, head_dim).transpose(1, 2, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert a Gemma4 checkpoint to paz")
    add = parser.add_argument
    add("--checkpoint_dir", required=True)
    add("--output_dir", required=True)
    add("--model_name", default="gemma4_12b")
    args = parser.parse_args()
    convert(args.checkpoint_dir, args.output_dir, args.model_name)
