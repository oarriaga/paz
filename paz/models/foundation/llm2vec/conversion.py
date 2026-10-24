"""Converts Llama-3-8B-Instruct plus the LLM2Vec LoRA adapters to paz weights.

Development tool. The base checkpoint is read through torch so bfloat16
tensors load; each LoRA adapter is merged as PEFT does (``W + B @ A *
alpha / r`` accumulated in float32, rounded to bfloat16), first the MNTP
adapter, then the supervised one. Every base and adapter tensor must be
consumed except the unused ``lm_head``.

    python -m paz.models.foundation.llm2vec.conversion \\
        --base_dir <snapshot of NousResearch/Meta-Llama-3-8B-Instruct> \\
        --adapter_dirs <mntp snapshot> <supervised snapshot> \\
        --output_dir ~/.keras/paz/models/llm2vec
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from paz.models.foundation.llm2vec import WEIGHTS_NAME
from paz.models.foundation.llm2vec import model as encoder


def convert(base_dir, adapter_dirs, output_dir):
    base = open_base(base_dir)
    adapters = [open_adapter(directory) for directory in adapter_dirs]
    with jax.default_device(jax.devices("cpu")[0]):
        model = encoder.build()
        used = transfer(model, base, adapters)
    audit(base, adapters, used)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_dir / WEIGHTS_NAME))
    report = "\n".join(f"{key} -> {target}" for key, target in sorted(used))
    (output_dir / "llm2vec_llama3_keymap.txt").write_text(report + "\n")
    return model


def open_base(base_dir):
    from safetensors import safe_open
    base_dir = Path(base_dir)
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    handles = {}
    for key, filename in index["weight_map"].items():
        if filename not in handles:
            handles[filename] = safe_open(str(base_dir / filename), "pt")
    weight_map = index["weight_map"]
    return {key: handles[filename] for key, filename in weight_map.items()}


def open_adapter(adapter_dir):
    from safetensors import safe_open
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    handle = safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt")
    return {key: handle for key in handle.keys()}, scale


def read(source, key):
    return source[key].get_tensor(key).float().numpy()


def transfer(model, base, adapters):
    used = set()

    def read_base(key, target):
        used.add((key, target))
        return read(base, key)

    def read_merged(relative, target):
        return merge_adapters(base, adapters, relative, used, target)

    embeddings = model.get_layer("token_embedding").embeddings
    key = "model.embed_tokens.weight"
    embeddings.assign(read_base(key, "token_embedding"))
    scale = model.get_layer("final_norm").scale
    scale.assign(read_base("model.norm.weight", "final_norm"))
    for layer in range(count_layers(base)):
        transfer_layer(model, read_base, read_merged, layer)
    return used


def transfer_layer(model, read_base, read_merged, layer):
    source, target = f"model.layers.{layer}", f"layer_{layer}"
    for name, key in (("attention_norm", "input_layernorm"),
                      ("feedforward_norm", "post_attention_layernorm")):
        layer_name = f"{target}_{name}"
        scale = model.get_layer(layer_name).scale
        scale.assign(read_base(f"{source}.{key}.weight", layer_name))
    projections = (("attention_query", "self_attn.q_proj"),
                   ("attention_key", "self_attn.k_proj"),
                   ("attention_value", "self_attn.v_proj"),
                   ("attention_output", "self_attn.o_proj"),
                   ("feedforward_gate", "mlp.gate_proj"),
                   ("feedforward_up", "mlp.up_proj"),
                   ("feedforward_down", "mlp.down_proj"))
    for name, key in projections:
        layer_name = f"{target}_{name}"
        kernel = model.get_layer(layer_name).kernel
        weight = read_merged(f"layers.{layer}.{key}", layer_name)
        kernel.assign(weight.T.reshape(kernel.shape))


def merge_adapters(base, adapters, relative, used, target):
    """Base weight with every LoRA delta folded in, rounded like PEFT."""
    key = f"model.{relative}.weight"
    used.add((key, target))
    weight = read(base, key)
    for adapter, scale in adapters:
        prefix = f"base_model.model.{relative}"
        down = read(adapter, f"{prefix}.lora_A.weight")
        up = read(adapter, f"{prefix}.lora_B.weight")
        used.add((f"{prefix}.lora_A.weight", target))
        used.add((f"{prefix}.lora_B.weight", target))
        weight = round_to_bfloat16(weight + (up @ down) * scale)
    return weight


def round_to_bfloat16(weight):
    return np.asarray(jax.numpy.asarray(weight, "bfloat16"), "float32")


def count_layers(base):
    prefix = "model.layers."
    layers = set()
    for key in base:
        if key.startswith(prefix):
            layers.add(key[len(prefix):].split(".")[0])
    return len(layers)


def audit(base, adapters, used):
    consumed = {key for key, _ in used}
    # The encoder never projects to the vocabulary.
    expected = set(base) - {"lm_head.weight"}
    for adapter, _ in adapters:
        expected.update(adapter)
    unused = sorted(expected - consumed)
    if unused:
        raise ValueError(f"Unmapped checkpoint tensors: {unused[:10]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--adapter_dirs", nargs=2, required=True)
    parser.add_argument("--output_dir", default="~/.keras/paz/models/llm2vec")
    arguments = parser.parse_args()
    convert(arguments.base_dir, arguments.adapter_dirs,
            Path(arguments.output_dir).expanduser())
