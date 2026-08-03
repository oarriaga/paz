import pytest
import torch
import numpy as np
from keras import Input, Model
import sys
import os

try:
    from ms_deform_attn import materialize_ms_deform_attn, run_ms_deform_attn
except ImportError:
    from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.ms_deform_attn import (  # fmt: skip
        materialize_ms_deform_attn,
        run_ms_deform_attn,
    )

try:
    from rfdetr import (
        RFDETRSmall,
        RFDETRMedium,
        RFDETRNano,
        RFDETRLarge,
    )
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rfdetr_root = os.path.abspath(os.path.join(
        current_dir,
        '../../../../../../examples/rf-detr_original_pytorch_implementation',
    ))
    sys.path.append(rfdetr_root)
    from rfdetr import (
        RFDETRSmall,
        RFDETRMedium,
        RFDETRNano,
        RFDETRLarge,
    )


from transformer_weights_porting_utils import (
    to_numpy,
    to_keras,
    get_ms_deform_attn_from_model
)

MS_DEFORM_PROJECTIONS = ("sampling_offsets", "attention_weights", "value_proj", "output_proj")  # fmt: skip


def extract_torch_ms_deform_attn(model_class):
    try:
        wrapper = model_class(pretrained=True)
        has_model = hasattr(wrapper, "model")
        has_nested = has_model and hasattr(wrapper.model, "model")
        torch_full_model = wrapper.model.model if has_nested else wrapper
        torch_full_model.eval()
    except Exception as error:
        pytest.fail(f"Failed to instantiate {model_class.__name__}: {error}")
    torch_attn = get_ms_deform_attn_from_model(torch_full_model)
    if torch_attn is None:
        pytest.fail(f"Could not locate MSDeformAttn in {model_class.__name__}")
    return torch_attn.cpu()


def build_keras_ms_deform_model(d_model, n_levels, n_heads, n_points):
    query_in = Input(shape=(None, d_model), name="query")
    memory_in = Input(shape=(None, d_model), name="memory")
    outputs = materialize_ms_deform_attn(query_in, memory_in, d_model, n_levels, n_heads, n_points, "msda")  # fmt: skip
    return Model([query_in, memory_in], outputs, name="ms_deform_attn")


def build_ms_deform_inputs(d_model, n_levels):
    batch_size, Len_q, Len_in = 1, 10, 20
    query_np = np.random.randn(batch_size, Len_q, d_model).astype(np.float32)
    ref_points_np = np.random.rand(batch_size, Len_q, n_levels, 4).astype(np.float32)  # fmt: skip
    input_flatten_np = np.random.randn(batch_size, Len_in, d_model).astype(np.float32)  # fmt: skip
    input_spatial_shapes_np = np.array([[5, 4]], dtype=np.int32)
    if n_levels > 1:
        # Adjust spatial shapes and total length for multiple levels
        input_spatial_shapes_np = np.array([[5, 4]] * n_levels, dtype=np.int32)
        Len_in = 20 * n_levels
        input_flatten_np = np.random.randn(batch_size, Len_in, d_model).astype(np.float32)  # fmt: skip
    print(f"Test Input Sizes: Len_q={Len_q}, Len_in={Len_in}")
    return query_np, ref_points_np, input_flatten_np, input_spatial_shapes_np


def transfer_ms_deform_weights(torch_attn, keras_model):
    with torch.no_grad():
        for part in MS_DEFORM_PROJECTIONS:
            layer = keras_model.get_layer(f"msda_{part}")
            module = getattr(torch_attn, part)
            layer.kernel.assign(to_keras(module.weight.T.cpu().numpy()))
            layer.bias.assign(to_keras(module.bias.cpu().numpy()))


def run_torch_ms_deform_attn(torch_attn, probe):
    query_np, ref_points_np, input_flatten_np, spatial_shapes_np = probe
    t_query = torch.from_numpy(query_np)
    t_ref_points = torch.from_numpy(ref_points_np)
    t_input_flatten = torch.from_numpy(input_flatten_np)
    t_spatial_shapes = torch.from_numpy(spatial_shapes_np).long()
    # Level start indices for the reference implementation
    lens = t_spatial_shapes[:, 0] * t_spatial_shapes[:, 1]
    starts = torch.cat((torch.tensor([0]), torch.cumsum(lens, 0)[:-1]))
    args = (t_query, t_ref_points, t_input_flatten, t_spatial_shapes)
    with torch.no_grad():
        outputs = torch_attn(*args, starts.long())
    return outputs


@pytest.mark.parametrize(
    "model_class", [RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge]
)
def test_rfdetr_ms_deform_attn_real_weights(model_class):
    print(f"\nTesting MSDeformAttn parity for {model_class.__name__}...")
    torch_attn = extract_torch_ms_deform_attn(model_class)
    d_model, n_levels = torch_attn.d_model, torch_attn.n_levels
    n_heads, n_points = torch_attn.n_heads, torch_attn.n_points
    print(f"Config: d_model={d_model}, n_levels={n_levels}, n_heads={n_heads}, n_points={n_points}")  # fmt: skip
    keras_model = build_keras_ms_deform_model(d_model, n_levels, n_heads, n_points)  # fmt: skip
    probe = build_ms_deform_inputs(d_model, n_levels)
    transfer_ms_deform_weights(torch_attn, keras_model)
    out_torch = run_torch_ms_deform_attn(torch_attn, probe)
    args = (keras_model, to_keras(probe[0]), to_keras(probe[1]))
    tail = (probe[3], None, n_levels, n_heads, n_points, "msda")
    out_keras = run_ms_deform_attn(*args, to_keras(probe[2]), *tail)
    diff = np.abs(to_numpy(out_torch) - to_numpy(out_keras))
    print(f"Max diff for {model_class.__name__}: {diff.max()}")
    assert np.allclose(
        to_numpy(out_torch), to_numpy(out_keras), atol=1e-5, rtol=1e-5
    ), f"Mismatch for {model_class.__name__}! Max diff: {diff.max()}"


if __name__ == "__main__":
    pytest.main([__file__])
