import numpy as np
import pytest

safetensors = pytest.importorskip("safetensors.numpy")

from paz.models.foundation.kimodo import conversion
from paz.models.foundation.kimodo import model as denoiser_model
from paz.models.foundation.kimodo.model_test import TINY, build_unit_stats
from paz.models.foundation.kimodo.model_test import build_inputs


def build_fake_checkpoint(path, hidden, ff, num_layers, text_dim):
    rng = np.random.default_rng(0)
    tensors = {}

    def dense(key, out_dim, in_dim):
        tensors[f"{key}.weight"] = rng.normal(size=(out_dim, in_dim))
        tensors[f"{key}.bias"] = rng.normal(size=(out_dim,))

    def norm(key):
        tensors[f"{key}.weight"] = rng.normal(size=(hidden,))
        tensors[f"{key}.bias"] = rng.normal(size=(hidden,))

    stages = (("root_model", 834, 5), ("body_model", 833, 412))
    for stage, in_dim, out_dim in stages:
        prefix = conversion.PREFIX + stage
        dense(f"{prefix}.embed_text", hidden, text_dim)
        dense(f"{prefix}.embed_timestep.time_embed.0", hidden, hidden)
        dense(f"{prefix}.embed_timestep.time_embed.2", hidden, hidden)
        dense(f"{prefix}.linear_first_heading_angle", hidden, 2)
        dense(f"{prefix}.input_linear", hidden, in_dim)
        dense(f"{prefix}.output_linear", out_dim, hidden)
        for layer in range(num_layers):
            key = f"{prefix}.seqTransEncoder.layers.{layer}"
            tensors[f"{key}.self_attn.in_proj_weight"] = rng.normal(
                size=(3 * hidden, hidden))
            tensors[f"{key}.self_attn.in_proj_bias"] = rng.normal(
                size=(3 * hidden,))
            dense(f"{key}.self_attn.out_proj", hidden, hidden)
            dense(f"{key}.linear1", ff, hidden)
            dense(f"{key}.linear2", hidden, ff)
            norm(f"{key}.norm1")
            norm(f"{key}.norm2")
    tensors = {k: (v * 0.1).astype("float32") for k, v in tensors.items()}
    safetensors.save_file(tensors, str(path))
    return tensors


def test_transfer_maps_every_tensor(tmp_path):
    from safetensors import safe_open
    path = tmp_path / "model.safetensors"
    args = (TINY["hidden_dim"], TINY["ff_dim"], TINY["num_layers"],
            TINY["text_dim"])
    tensors = build_fake_checkpoint(path, *args)
    checkpoint = safe_open(str(path), "numpy")
    model = denoiser_model.build(build_unit_stats(), **TINY)
    used = conversion.transfer(checkpoint, model)
    conversion.audit_keys(checkpoint, used)
    assert set(used) == set(tensors)
    key = conversion.PREFIX + "body_model.output_linear.weight"
    kernel = model.get_layer("body_output_projection").get_weights()[0]
    assert np.allclose(kernel, tensors[key].T)
    assert np.isfinite(np.asarray(model(build_inputs(4)))).all()


def test_audit_rejects_unmapped_tensors(tmp_path):
    from safetensors import safe_open
    path = tmp_path / "model.safetensors"
    args = (TINY["hidden_dim"], TINY["ff_dim"], TINY["num_layers"],
            TINY["text_dim"])
    tensors = build_fake_checkpoint(path, *args)
    tensors["denoiser.backbone.extra"] = np.zeros((1,), "float32")
    safetensors.save_file(tensors, str(path))
    checkpoint = safe_open(str(path), "numpy")
    model = denoiser_model.build(build_unit_stats(), **TINY)
    used = conversion.transfer(checkpoint, model)
    with pytest.raises(ValueError, match="extra"):
        conversion.audit_keys(checkpoint, used)


def test_shape_mismatch_fails(tmp_path):
    from safetensors import safe_open
    path = tmp_path / "model.safetensors"
    args = (TINY["hidden_dim"], TINY["ff_dim"], TINY["num_layers"],
            TINY["text_dim"])
    build_fake_checkpoint(path, *args)
    checkpoint = safe_open(str(path), "numpy")
    wrong = dict(TINY, hidden_dim=TINY["hidden_dim"] * 2)
    model = denoiser_model.build(build_unit_stats(), **wrong)
    with pytest.raises(ValueError):
        conversion.transfer(checkpoint, model)
