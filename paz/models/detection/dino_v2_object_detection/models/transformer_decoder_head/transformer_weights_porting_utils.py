from collections import namedtuple

import numpy as np
import pytest
import torch
from keras import Input, Model, ops, layers

from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (  # fmt: skip
    Transformer as KerasTransformer,
    apply_transformer,
    mlp,
)

ACTIVATION_NAMES = ("relu", "gelu", "glu")
CROSS_ATTENTION_PARTS = ("sampling_offsets", "attention_weights", "value_proj", "output_proj")  # fmt: skip
OUTPUT_NAMES = ("Hidden States", "References", "Memory TS", "Boxes TS")
PARITY_TOLERANCES = {"Hidden States": 1e-3, "References": 1e-4, "Memory TS": 1e-4, "Boxes TS": 1e-4}  # fmt: skip
PROBE_SIZE = 32

ParityProbe = namedtuple("ParityProbe", "sources masks positions query_feat refpoints")  # fmt: skip


def build_parity_mlp(d_model, name):
    inputs = Input((None, d_model), name=f"{name}_input")
    outputs = mlp(inputs, d_model, d_model, 4, 3, name)
    return Model(inputs, outputs, name=name)


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        result = t.detach().cpu().numpy()
    elif hasattr(t, "numpy"):
        result = t.numpy()
    else:
        result = np.array(t)
    return result


def to_torch(x):
    return torch.tensor(x, dtype=torch.float32)


def to_keras(x):
    return ops.convert_to_tensor(x, dtype="float32")


def get_ms_deform_attn_from_model(model):
    try:
        result = unwrap_model(model).transformer.decoder.layers[0].cross_attn
    except AttributeError:
        result = find_cross_attention_module(model)
    return result


def unwrap_model(model):
    inner = getattr(model, "model", model)
    return getattr(inner, "model", inner)


def find_cross_attention_module(model):
    found = None
    for name, module in model.named_modules():
        parts = ("cross_attn" in name, "decoder" in name, "layers.0" in name)
        if all(parts):
            found = module
            break
    return found


def extract_pt_transformer(model_class, config=None):
    if config is not None:
        wrapper = model_class(config=config)
    else:
        wrapper = model_class(pretrained=True)
    wrapper.model.model.eval()
    wrapper.model.model.cpu()
    return wrapper.model.model.transformer


def transfer_mha(t_mha, k_mha, d_model, sa_nhead):
    projections = (k_mha.query_dense, k_mha.key_dense, k_mha.value_dense)
    for index, dense in enumerate(projections):
        start = index * d_model
        args = (t_mha, dense, start, start + d_model)
        transfer_qkv_weights(*args, d_model, sa_nhead)
    transfer_output_projection(t_mha, k_mha.output_dense, d_model, sa_nhead)


def transfer_qkv_weights(t_mha, dense, start, end, d_model, sa_nhead):
    # in_proj_weight is the fused Q|K|V stack; each slice is transposed and
    # reshaped into the Keras (d_model, num_heads, head_dim) kernel layout.
    kernel = to_keras(t_mha.in_proj_weight[start:end, :].T.cpu().numpy())
    head_dim = d_model // sa_nhead
    dense.kernel.assign(ops.reshape(kernel, (d_model, sa_nhead, head_dim)))
    bias = to_keras(t_mha.in_proj_bias[start:end].cpu().numpy())
    dense.bias.assign(ops.reshape(bias, (sa_nhead, head_dim)))


def transfer_output_projection(t_mha, dense, d_model, sa_nhead):
    kernel = to_keras(t_mha.out_proj.weight.T.cpu().numpy())
    head_dim = d_model // sa_nhead
    dense.kernel.assign(ops.reshape(kernel, (sa_nhead, head_dim, d_model)))
    dense.bias.assign(to_keras(t_mha.out_proj.bias.cpu().numpy()))


def transfer_layernorm(t_norm, k_norm):
    k_norm.gamma.assign(to_keras(t_norm.weight.cpu().numpy()))
    k_norm.beta.assign(to_keras(t_norm.bias.cpu().numpy()))


def transfer_dense(t_linear, k_dense):
    k_dense.kernel.assign(to_keras(t_linear.weight.T.cpu().numpy()))
    k_dense.bias.assign(to_keras(t_linear.bias.cpu().numpy()))


def transfer_mlp(t_mlp, k_mlp):
    for index, t_layer in enumerate(t_mlp.layers):
        transfer_dense(t_layer, k_mlp.get_layer(f"{k_mlp.name}_dense_{index}"))


def transfer_transformer_weights(pt_transformer, keras_transformer, d_model, sa_nhead):  # fmt: skip
    with torch.no_grad():
        if pt_transformer.two_stage:
            transfer_encoder_outputs(pt_transformer, keras_transformer)
        for index, t_layer in enumerate(pt_transformer.decoder.layers):
            args = (t_layer, keras_transformer, index)
            transfer_decoder_layer(*args, d_model, sa_nhead)
        transfer_reference_point_head(pt_transformer, keras_transformer)
        norm = keras_transformer.get_layer("decoder_norm")
        transfer_layernorm(pt_transformer.decoder.norm, norm)


def transfer_encoder_outputs(pt_transformer, keras_transformer):
    num_groups = keras_transformer.transformer_config["group_detr"]
    for group in range(num_groups):
        dense = keras_transformer.get_layer(f"enc_output_{group}")
        transfer_dense(pt_transformer.enc_output[group], dense)
        norm = keras_transformer.get_layer(f"enc_output_norm_{group}")
        transfer_layernorm(pt_transformer.enc_output_norm[group], norm)


def transfer_decoder_layer(t_layer, keras_transformer, index, d_model, sa_nhead):  # fmt: skip
    name = f"decoder_layer_{index}"
    get_layer = keras_transformer.get_layer
    transfer_mha(t_layer.self_attn, get_layer(f"{name}_self_attn"), d_model, sa_nhead)  # fmt: skip
    transfer_layernorm(t_layer.norm1, get_layer(f"{name}_norm1"))
    args = (t_layer.cross_attn, keras_transformer, f"{name}_cross_attn")
    transfer_cross_attention(*args)
    transfer_layernorm(t_layer.norm2, get_layer(f"{name}_norm2"))
    transfer_dense(t_layer.linear1, get_layer(f"{name}_linear1"))
    transfer_dense(t_layer.linear2, get_layer(f"{name}_linear2"))
    transfer_layernorm(t_layer.norm3, get_layer(f"{name}_norm3"))


def transfer_cross_attention(t_cross, keras_transformer, prefix):
    for part in CROSS_ATTENTION_PARTS:
        dense = keras_transformer.get_layer(f"{prefix}_{part}")
        transfer_dense(getattr(t_cross, part), dense)


def transfer_reference_point_head(pt_transformer, keras_transformer):
    for index, t_layer in enumerate(pt_transformer.decoder.ref_point_head.layers):  # fmt: skip
        name = f"decoder_ref_point_head_dense_{index}"
        transfer_dense(t_layer, keras_transformer.get_layer(name))


def build_parity_heads(pt_transformer, d_model):
    enc_out_class_embed = None
    enc_out_bbox_embed = None
    bbox_embed = None
    if hasattr(pt_transformer, "enc_out_class_embed"):
        heads = pt_transformer.enc_out_class_embed
        enc_out_class_embed = [build_parity_dense(head, d_model) for head in heads]  # fmt: skip
    if hasattr(pt_transformer, "enc_out_bbox_embed"):
        count = len(pt_transformer.enc_out_bbox_embed)
        enc_out_bbox_embed = [build_parity_mlp(d_model, f"enc_out_bbox_embed_{index}") for index in range(count)]  # fmt: skip
    if getattr(pt_transformer.decoder, "bbox_embed", None) is not None:
        bbox_embed = build_parity_mlp(d_model, "bbox_embed")
    return bbox_embed, enc_out_class_embed, enc_out_bbox_embed


def build_parity_dense(torch_head, d_model):
    dense = layers.Dense(torch_head.out_features)
    dense.build((None, d_model))
    return dense


def transfer_parity_heads(pt_transformer, bbox_embed, enc_out_class_embed, enc_out_bbox_embed):  # fmt: skip
    with torch.no_grad():
        args = (pt_transformer, "enc_out_class_embed", enc_out_class_embed)
        transfer_head_list(*args, transfer_dense)
        args = (pt_transformer, "enc_out_bbox_embed", enc_out_bbox_embed)
        transfer_head_list(*args, transfer_mlp)
        if bbox_embed is not None:
            transfer_mlp(pt_transformer.decoder.bbox_embed, bbox_embed)


def transfer_head_list(pt_transformer, attribute, keras_heads, transfer):
    if keras_heads is not None:
        torch_heads = getattr(pt_transformer, attribute)
        for torch_head, keras_head in zip(torch_heads, keras_heads):
            transfer(torch_head, keras_head)


def read_model_dim(pt_transformer):
    if hasattr(pt_transformer, "d_model"):
        dim = pt_transformer.d_model
    else:
        dim = pt_transformer.decoder.layers[0].linear1.in_features
    return dim


def read_two_stage(pt_transformer):
    heads = getattr(pt_transformer, "enc_out_class_embed", None)
    return heads is not None and len(heads) > 0


def read_sampling_points(cross_attention):
    out_features = cross_attention.sampling_offsets.weight.shape[0]
    divisor = cross_attention.n_heads * cross_attention.n_levels * 2
    derived = out_features // divisor
    if derived != cross_attention.n_points:
        print(f"WARNING: MSDeformAttn.n_points ({cross_attention.n_points}) differs from weight shape derived ({derived}). Using derived value.")  # fmt: skip
    return derived


def read_activation(decoder_layer):
    activation = getattr(decoder_layer, "activation", None)
    name = getattr(activation, "__name__", "")
    found = [known for known in ACTIVATION_NAMES if known in name]
    return found[0] if found else "relu"


def read_torch_transformer_config(pt_transformer):
    config = read_transformer_shape(pt_transformer)
    config.update(read_transformer_flags(pt_transformer))
    return config


def read_transformer_shape(pt_transformer):
    layer = pt_transformer.decoder.layers[0]
    cross = layer.cross_attn
    keys = ("d_model", "sa_nhead", "ca_nhead", "num_queries", "num_decoder_layers", "dim_feedforward", "dropout", "num_feature_levels", "dec_n_points")  # fmt: skip
    values = (read_model_dim(pt_transformer), layer.self_attn.num_heads, cross.n_heads, pt_transformer.num_queries, len(pt_transformer.decoder.layers), layer.linear1.out_features, layer.dropout1.p, cross.n_levels, read_sampling_points(cross))  # fmt: skip
    return dict(zip(keys, values))


def read_transformer_flags(pt_transformer):
    decoder = pt_transformer.decoder
    layer = decoder.layers[0]
    keys = ("return_intermediate_dec", "group_detr", "two_stage", "bbox_reparam", "lite_refpoint_refine", "activation", "normalize_before")  # fmt: skip
    values = (getattr(decoder, "return_intermediate", True), getattr(pt_transformer, "group_detr", 1), read_two_stage(pt_transformer), getattr(pt_transformer, "bbox_reparam", False), getattr(decoder, "lite_refpoint_refine", False), read_activation(layer), getattr(layer, "normalize_before", False))  # fmt: skip
    return dict(zip(keys, values))


def build_parity_probe(config):
    d_model = config["d_model"]
    sources, masks, positions = [], [], []
    for level in range(config["num_feature_levels"]):
        extent = max(1, PROBE_SIZE // (2**level))
        sources.append(np.random.randn(1, d_model, extent, extent).astype(np.float32))  # fmt: skip
        masks.append(np.zeros((1, extent, extent), dtype=bool))
        positions.append(np.random.randn(1, d_model, extent, extent).astype(np.float32))  # fmt: skip
    num_queries = config["num_queries"]
    query_feat = np.random.randn(num_queries, d_model).astype(np.float32)
    refpoints = np.random.randn(num_queries, 4).astype(np.float32)
    return ParityProbe(sources, masks, positions, query_feat, refpoints)


def run_torch_transformer(pt_transformer, probe):
    sources = [torch.tensor(x) for x in probe.sources]
    masks = [torch.tensor(x) for x in probe.masks]
    positions = [torch.tensor(x) for x in probe.positions]
    refpoints = torch.tensor(probe.refpoints)
    query_feat = torch.tensor(probe.query_feat)
    with torch.no_grad():
        outputs = pt_transformer(sources, masks, positions, refpoints, query_feat)  # fmt: skip
    return outputs


def run_keras_transformer(keras_transformer, heads, probe):
    # apply_transformer consumes NHWC feature maps
    sources = [to_keras(np.transpose(x, (0, 2, 3, 1))) for x in probe.sources]
    masks = [to_keras(x) for x in probe.masks]
    positions = [to_keras(np.transpose(x, (0, 2, 3, 1))) for x in probe.positions]  # fmt: skip
    args = (keras_transformer, sources, masks, positions, *heads)
    tail = (to_keras(probe.query_feat), to_keras(probe.refpoints), False)
    return apply_transformer(*args, *tail)


def compare_parity_arrays(name, torch_value, keras_value):
    failure = None
    if torch_value.shape != keras_value.shape:
        print(f"  {name} Shape Mismatch: PT {torch_value.shape} vs Keras {keras_value.shape}")  # fmt: skip
        failure = f"{name} shape mismatch"
    else:
        difference = np.abs(torch_value - keras_value)
        mean_difference = difference.mean()
        print(f"  {name} Mean Diff: {mean_difference:.6f} (Max Diff: {difference.max():.6f})")  # fmt: skip
        tolerance = PARITY_TOLERANCES.get(name, 1e-4)
        if mean_difference > tolerance:
            failure = f"{name} mean diff {mean_difference} > {tolerance}"
    return failure


def compare_parity_output(name, torch_output, keras_output):
    torch_value = to_numpy(torch_output)
    failure = None
    if keras_output is None:
        if torch_value is not None:
            print(f"  {name} Mismatch: PT is {torch_value.shape}, Keras is None")  # fmt: skip
            failure = f"{name} is None in Keras"
    else:
        failure = compare_parity_arrays(name, torch_value, to_numpy(keras_output))  # fmt: skip
    return failure


def collect_parity_failures(torch_outputs, keras_outputs):
    failures = []
    for index, torch_output in enumerate(torch_outputs):
        args = (OUTPUT_NAMES[index], torch_output, keras_outputs[index])
        failure = compare_parity_output(*args)
        if failure is not None:
            failures.append(failure)
    return failures


def verify_transformer_parity(pt_transformer, variant_name):
    config = read_torch_transformer_config(pt_transformer)
    print(f"Config: {config}")
    keras_transformer = KerasTransformer(**config)
    heads = build_parity_heads(pt_transformer, config["d_model"])
    probe = build_parity_probe(config)
    args = (pt_transformer, keras_transformer, config["d_model"])
    transfer_transformer_weights(*args, config["sa_nhead"])
    transfer_parity_heads(pt_transformer, *heads)
    torch_outputs = run_torch_transformer(pt_transformer, probe)
    keras_outputs = run_keras_transformer(keras_transformer, heads, probe)
    failures = collect_parity_failures(torch_outputs, keras_outputs)
    if failures:
        pytest.fail(f"Parity check failed: {failures}")
    print(f"RFDETR {variant_name} Transformer parity PASSED")
