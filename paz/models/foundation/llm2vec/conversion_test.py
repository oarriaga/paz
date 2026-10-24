import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from paz.models.foundation.llm2vec import conversion  # noqa: E402
from paz.models.foundation.llm2vec import model as encoder  # noqa: E402
from paz.models.foundation.llm2vec.model_test import TINY  # noqa: E402

RANK = 2


def write_base(directory, rng):
    hidden, kv = TINY["hidden_dim"], TINY["num_kv_heads"] * TINY["head_dim"]
    shapes = {"model.embed_tokens.weight": (TINY["vocab_size"], hidden),
              "model.norm.weight": (hidden,),
              "lm_head.weight": (TINY["vocab_size"], hidden)}
    for layer in range(TINY["num_layers"]):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (hidden, hidden)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (kv, hidden)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (kv, hidden)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (hidden, hidden)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (TINY["ff_dim"], hidden)
        shapes[f"{prefix}.mlp.up_proj.weight"] = (TINY["ff_dim"], hidden)
        shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden, TINY["ff_dim"])
    tensors = {key: torch.tensor(rng.normal(size=shape), dtype=torch.bfloat16)
               for key, shape in shapes.items()}
    names = sorted(tensors)
    half = len(names) // 2
    files = {"a.safetensors": names[:half], "b.safetensors": names[half:]}
    for filename, keys in files.items():
        shard = {key: tensors[key] for key in keys}
        save_file(shard, str(directory / filename))
    weight_map = {key: filename for filename, keys in files.items()
                  for key in keys}
    index = {"weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    return {key: value.float().numpy() for key, value in tensors.items()}


def write_adapter(directory, rng, alpha):
    directory.mkdir()
    tensors = {}
    for layer in range(TINY["num_layers"]):
        prefix = f"base_model.model.layers.{layer}"
        for key, shape in iter_lora_shapes():
            tensors[f"{prefix}.{key}.lora_A.weight"] = rng.normal(
                size=(RANK, shape[1]))
            tensors[f"{prefix}.{key}.lora_B.weight"] = rng.normal(
                size=(shape[0], RANK))
    tensors = {key: torch.tensor(value, dtype=torch.float32)
               for key, value in tensors.items()}
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    config = {"r": RANK, "lora_alpha": alpha}
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return {key: value.numpy() for key, value in tensors.items()}


def iter_lora_shapes():
    hidden, kv = TINY["hidden_dim"], TINY["num_kv_heads"] * TINY["head_dim"]
    return (("self_attn.q_proj", (hidden, hidden)),
            ("self_attn.k_proj", (kv, hidden)),
            ("self_attn.v_proj", (kv, hidden)),
            ("self_attn.o_proj", (hidden, hidden)),
            ("mlp.gate_proj", (TINY["ff_dim"], hidden)),
            ("mlp.up_proj", (TINY["ff_dim"], hidden)),
            ("mlp.down_proj", (hidden, TINY["ff_dim"])))


def test_transfer_merges_both_adapters(tmp_path):
    rng = np.random.default_rng(0)
    base = write_base(tmp_path, rng)
    first = write_adapter(tmp_path / "mntp", rng, alpha=4)
    second = write_adapter(tmp_path / "supervised", rng, alpha=2)
    model = encoder.build(**TINY)
    adapters = [conversion.open_adapter(tmp_path / "mntp"),
                conversion.open_adapter(tmp_path / "supervised")]
    used = conversion.transfer(model, conversion.open_base(tmp_path), adapters)
    conversion.audit(conversion.open_base(tmp_path), adapters, used)
    key = "model.layers.1.mlp.down_proj.weight"
    expected = base[key]
    for adapter, scale in ((first, 2.0), (second, 1.0)):
        prefix = "base_model.model.layers.1.mlp.down_proj"
        up = adapter[f"{prefix}.lora_B.weight"]
        delta = up @ adapter[f"{prefix}.lora_A.weight"]
        expected = conversion.round_to_bfloat16(expected + delta * scale)
    kernel = model.get_layer("layer_1_feedforward_down").kernel
    assert np.allclose(np.asarray(kernel, "float32"), expected.T, atol=1e-6)
    query = model.get_layer("layer_0_attention_query").kernel
    assert query.shape == (TINY["hidden_dim"], TINY["num_heads"],
                           TINY["head_dim"])
    assert np.allclose(np.asarray(model.get_layer("final_norm").scale,
                                  "float32"), base["model.norm.weight"])


def test_audit_rejects_unused_base_tensors(tmp_path):
    rng = np.random.default_rng(1)
    write_base(tmp_path, rng)
    write_adapter(tmp_path / "mntp", rng, alpha=4)
    base = conversion.open_base(tmp_path)
    adapters = [conversion.open_adapter(tmp_path / "mntp")]
    with pytest.raises(ValueError, match="Unmapped"):
        conversion.audit(base, adapters, set())
