import math
from collections import namedtuple
from functools import partial

import keras
from keras import Input, Model, layers, ops

from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.ms_deform_attn import (  # fmt: skip
    materialize_ms_deform_attn,
    run_ms_deform_attn,
)

REFERENCE_POINT_HEAD = "decoder_ref_point_head"
DECODER_NORM = "decoder_norm"
DECODER_LAYER = "decoder_layer_{}"
ENCODER_OUTPUT = "enc_output_{}"
ENCODER_OUTPUT_NORM = "enc_output_norm_{}"
SINE_TEMPERATURE = 10000

ACTIVATIONS = {
    "relu": keras.activations.relu,
    "gelu": keras.activations.gelu,
    "glu": keras.activations.glu,
}

DecoderReference = namedtuple("DecoderReference", ["points", "query_pos"])
DecoderQueries = namedtuple("DecoderQueries", ["target", "refpoints"])
EncoderProposals = namedtuple("EncoderProposals", ["refpoints", "memory", "boxes"])  # fmt: skip


# input_dim mirrors the upstream PyTorch MLP(input_dim, hidden_dim, ...)
# positional signature that the porter parity tests build against.
def mlp(x, input_dim, hidden_dim, output_dim, num_layers, name):
    dims = [hidden_dim] * (num_layers - 1) + [output_dim]
    for index, dim in enumerate(dims):
        x = layers.Dense(dim, name=f"{name}_dense_{index}")(x)
        if index < num_layers - 1:
            x = ops.relu(x)
    return x


def apply_mlp(model, x, num_layers, name):
    for index in range(num_layers):
        x = model.get_layer(f"{name}_dense_{index}")(x)
        if index < num_layers - 1:
            x = ops.relu(x)
    return x


def embed_position_sine(positions, dim=128):
    scale = 2 * math.pi
    frequencies = build_sine_frequencies(dim, positions.dtype)
    embed = partial(embed_coordinate_sine, positions, frequencies, scale)
    parts = [embed(1), embed(0)]
    if ops.shape(positions)[-1] == 4:
        parts = parts + [embed(2), embed(3)]
    return ops.concatenate(parts, axis=2)


def build_sine_frequencies(dim, dtype):
    steps = ops.arange(dim, dtype=dtype)
    return SINE_TEMPERATURE ** (2 * ops.floor(steps / 2) / dim)


def embed_coordinate_sine(positions, frequencies, scale, axis):
    scaled = ops.expand_dims(positions[:, :, axis] * scale, axis=-1)
    scaled = scaled / frequencies
    sines = ops.sin(scaled[:, :, 0::2])
    cosines = ops.cos(scaled[:, :, 1::2])
    interleaved = ops.stack([sines, cosines], axis=-1)
    shape = (ops.shape(scaled)[0], ops.shape(scaled)[1], -1)
    return ops.reshape(interleaved, shape)


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, unsigmoid=True):  # fmt: skip
    args = (memory, memory_padding_mask, spatial_shapes)
    proposals = build_proposal_grid(*args)
    validity = compute_proposal_validity(proposals)
    if unsigmoid:
        proposals = unsigmoid_proposals(proposals, memory_padding_mask, validity)  # fmt: skip
    else:
        proposals = zero_where_invalid(proposals, memory_padding_mask, validity)
    output = zero_where_invalid(memory, memory_padding_mask, validity)
    return ops.cast(output, memory.dtype), ops.cast(proposals, memory.dtype)


def build_proposal_grid(memory, padding_mask, spatial_shapes):
    batch = ops.shape(memory)[0]
    proposals = []
    start = 0
    for level in range(count_levels(spatial_shapes)):
        height, width = read_level_shape(spatial_shapes, level)
        extent = compute_valid_extent(padding_mask, batch, start, height, width)
        args = (batch, level, height, width)
        proposals.append(build_level_proposal(*args, *extent))
        start = start + height * width
    return ops.concatenate(proposals, axis=1)


def count_levels(spatial_shapes):
    if isinstance(spatial_shapes, list):
        num_levels = len(spatial_shapes)
    else:
        num_levels = ops.shape(spatial_shapes)[0]
    return num_levels


def read_level_shape(spatial_shapes, level):
    if isinstance(spatial_shapes, list):
        height, width = spatial_shapes[level]
    else:
        height = spatial_shapes[level][0]
        width = spatial_shapes[level][1]
    return height, width


def compute_valid_extent(padding_mask, batch, start, height, width):
    if padding_mask is None:
        valid_height = ops.full((batch,), ops.cast(height, "float32"))
        valid_width = ops.full((batch,), ops.cast(width, "float32"))
    else:
        window = padding_mask[:, start : (start + height * width)]
        window = ops.reshape(window, (batch, height, width, 1))
        unmasked = ops.logical_not(ops.cast(window, "bool"))
        valid_height = ops.sum(ops.cast(unmasked[:, :, 0, 0], "float32"), axis=1)  # fmt: skip
        valid_width = ops.sum(ops.cast(unmasked[:, 0, :, 0], "float32"), axis=1)
    return valid_height, valid_width


def build_level_proposal(batch, level, height, width, valid_height, valid_width):  # fmt: skip
    rows = ops.linspace(0.0, ops.cast(height - 1, "float32"), int(height))
    columns = ops.linspace(0.0, ops.cast(width - 1, "float32"), int(width))
    row_mesh, column_mesh = ops.meshgrid(rows, columns, indexing="ij")
    grid = ops.expand_dims(ops.stack([column_mesh, row_mesh], axis=-1), axis=0)
    extent = ops.stack([valid_width, valid_height], axis=1)
    # The +0.5 centers each pixel inside the valid (non-padded) extent.
    centers = (grid + 0.5) / ops.reshape(extent, (batch, 1, 1, 2))
    # Default width/height grows with the level (coarser anchors deeper).
    sizes = ops.ones_like(centers) * 0.05 * (2.0**level)
    proposal = ops.concatenate([centers, sizes], axis=-1)
    return ops.reshape(proposal, (batch, -1, 4))


def compute_proposal_validity(proposals):
    inside = ops.logical_and(proposals > 0.01, proposals < 0.99)
    return ops.all(inside, axis=-1, keepdims=True)


def unsigmoid_proposals(proposals, padding_mask, validity):
    logits = ops.log(proposals / (1 - proposals))
    if padding_mask is not None:
        expanded = ops.expand_dims(padding_mask, axis=-1)
        logits = ops.where(expanded, float("inf"), logits)
    return ops.where(ops.logical_not(validity), float("inf"), logits)


def zero_where_invalid(tensor, padding_mask, validity):
    if padding_mask is not None:
        expanded = ops.expand_dims(padding_mask, axis=-1)
        tensor = ops.where(expanded, 0.0, tensor)
    return ops.where(ops.logical_not(validity), 0.0, tensor)


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


def apply_activation(x, activation):
    builder = ACTIVATIONS.get(activation) or keras.activations.get(activation)
    return builder(x)


def materialize_decoder_layer(query, memory, d_model, sa_nhead, ca_nhead, dim_feedforward, dropout, num_feature_levels, dec_n_points, name):  # fmt: skip
    keys = ("num_heads", "key_dim", "dropout", "name")
    values = (sa_nhead, d_model // sa_nhead, dropout, f"{name}_self_attn")
    self_attention = layers.MultiHeadAttention(**dict(zip(keys, values)))
    outputs = [self_attention(query=query, value=query, key=query)]
    outputs.append(build_layer_norm(f"{name}_norm1")(query))
    outputs += materialize_ms_deform_attn(query, memory, d_model, num_feature_levels, ca_nhead, dec_n_points, f"{name}_cross_attn")  # fmt: skip
    outputs.append(build_layer_norm(f"{name}_norm2")(query))
    hidden = layers.Dense(dim_feedforward, name=f"{name}_linear1")(query)
    outputs.append(layers.Dense(d_model, name=f"{name}_linear2")(hidden))
    outputs.append(build_layer_norm(f"{name}_norm3")(query))
    return outputs


def build_layer_norm(name):
    return layers.LayerNormalization(epsilon=1e-5, name=name)


def apply_decoder_layer(model, target, memory, d_model, group_detr, num_feature_levels, ca_nhead, dec_n_points, activation, dropout, query_pos, reference_points, spatial_shapes, memory_key_padding_mask, tgt_mask, training, name):  # fmt: skip
    args = (model, target, d_model, group_detr, query_pos, tgt_mask)
    target = apply_self_attention_block(*args, dropout, training, name)
    args = (model, target, memory, query_pos, reference_points, spatial_shapes)
    deform = (memory_key_padding_mask, num_feature_levels, ca_nhead, dec_n_points)  # fmt: skip
    target = apply_cross_attention_block(*args, *deform, dropout, training, name)  # fmt: skip
    return apply_feedforward_block(model, target, activation, dropout, training, name)  # fmt: skip


def apply_self_attention_block(model, target, d_model, group_detr, query_pos, target_mask, dropout, training, name):  # fmt: skip
    query = key = with_pos_embed(target, query_pos)
    value = target
    batch = ops.shape(target)[0]
    num_queries = ops.shape(target)[1]
    grouped = training and group_detr > 1
    if grouped:
        shape = (batch * group_detr, num_queries // group_detr, d_model)
        query, key, value = [ops.reshape(x, shape) for x in (query, key, value)]
    keys = ("query", "value", "key", "attention_mask", "training")
    values = (query, value, key, target_mask, training)
    attended = model.get_layer(f"{name}_self_attn")(**dict(zip(keys, values)))
    if grouped:
        attended = ops.reshape(attended, (batch, num_queries, d_model))
    dropped = layers.Dropout(dropout)(attended, training=training)
    return model.get_layer(f"{name}_norm1")(target + dropped)


def apply_cross_attention_block(model, target, memory, query_pos, reference_points, spatial_shapes, memory_key_padding_mask, num_feature_levels, ca_nhead, dec_n_points, dropout, training, name):  # fmt: skip
    query = with_pos_embed(target, query_pos)
    args = (model, query, reference_points, memory, spatial_shapes)
    deform = (memory_key_padding_mask, num_feature_levels, ca_nhead, dec_n_points)  # fmt: skip
    attended = run_ms_deform_attn(*args, *deform, f"{name}_cross_attn")
    dropped = layers.Dropout(dropout)(attended, training=training)
    return model.get_layer(f"{name}_norm2")(target + dropped)


def apply_feedforward_block(model, target, activation, dropout, training, name):
    hidden = model.get_layer(f"{name}_linear1")(target)
    hidden = apply_activation(hidden, activation)
    hidden = layers.Dropout(dropout)(hidden, training=training)
    projected = model.get_layer(f"{name}_linear2")(hidden)
    dropped = layers.Dropout(dropout)(projected, training=training)
    return model.get_layer(f"{name}_norm3")(target + dropped)


def apply_box_reparam(deltas, reference):
    centers = deltas[..., :2] * reference[..., 2:] + reference[..., :2]
    sizes = ops.exp(deltas[..., 2:]) * reference[..., 2:]
    return ops.concatenate([centers, sizes], axis=-1)


def refine_boxes(deltas, reference, bbox_reparam):
    if bbox_reparam:
        refined = apply_box_reparam(deltas, reference)
    else:
        refined = reference + deltas
    return refined


def apply_decoder(model, target, memory, config, bbox_embed, memory_key_padding_mask, refpoints_unsigmoid, spatial_shapes, valid_ratios, training):  # fmt: skip
    lite = config["lite_refpoint_refine"]
    refpoints = refpoints_unsigmoid
    reference = None
    if lite:
        reference = build_step_reference(model, config, refpoints, valid_ratios)  # fmt: skip
    keys =("d_model", "group_detr", "num_feature_levels", "ca_nhead", "dec_n_points", "activation", "dropout")  # fmt: skip
    kwargs = {key: config[key] for key in keys}
    kwargs.update(memory=memory, spatial_shapes=spatial_shapes, memory_key_padding_mask=memory_key_padding_mask, tgt_mask=None, training=training)  # fmt: skip
    apply_layer = partial(apply_decoder_layer, model, **kwargs)
    hidden, intermediate, trail = target, [], [refpoints]
    for layer in range(config["num_decoder_layers"]):
        if not lite:
            reference = build_step_reference(model, config, refpoints, valid_ratios)  # fmt: skip
        hidden = apply_layer(hidden, query_pos=reference.query_pos, reference_points=reference.points, name=DECODER_LAYER.format(layer))  # fmt: skip
        if not lite:
            refpoints = advance_refpoints(config, bbox_embed, hidden, refpoints, trail, layer)  # fmt: skip
        if config["return_intermediate_dec"]:
            intermediate.append(model.get_layer(DECODER_NORM)(hidden))
    return collect_decoder_outputs(model, hidden, intermediate, trail, refpoints, config)  # fmt: skip


def build_step_reference(model, config, refpoints, valid_ratios):
    if not config["bbox_reparam"]:
        refpoints = ops.sigmoid(refpoints)
    points = ops.expand_dims(refpoints[..., :4], axis=2)
    if valid_ratios is not None:
        ratios = ops.concatenate([valid_ratios, valid_ratios], axis=-1)
        points = points * ops.expand_dims(ratios, axis=1)
    sine = embed_position_sine(points[..., 0, :], config["d_model"] // 2)
    query_pos = apply_mlp(model, sine, 2, REFERENCE_POINT_HEAD)
    return DecoderReference(points, query_pos)


def advance_refpoints(config, bbox_embed, hidden, refpoints, trail, layer):
    if bbox_embed is not None:
        refined = refine_boxes(bbox_embed(hidden), refpoints, config["bbox_reparam"])  # fmt: skip
        if layer != config["num_decoder_layers"] - 1:
            trail.append(refined)
        refpoints = ops.stop_gradient(refined)
    return refpoints


def collect_decoder_outputs(model, hidden, intermediate, trail, refpoints, config):  # fmt: skip
    hidden = model.get_layer(DECODER_NORM)(hidden)
    if config["return_intermediate_dec"]:
        intermediate.pop()
        intermediate.append(hidden)
        result = ops.stack(intermediate), ops.stack(trail)
    else:
        result = ops.expand_dims(hidden, 0), ops.expand_dims(refpoints, 0)
    return result


def get_valid_ratio(mask):
    height = ops.shape(mask)[1]
    width = ops.shape(mask)[2]
    unmasked = ops.logical_not(ops.cast(mask, "bool"))
    valid_height = ops.sum(ops.cast(unmasked[:, :, 0], "float32"), axis=1)
    valid_width = ops.sum(ops.cast(unmasked[:, 0, :], "float32"), axis=1)
    ratio_height = valid_height / ops.cast(height, "float32")
    ratio_width = valid_width / ops.cast(width, "float32")
    return ops.stack([ratio_width, ratio_height], axis=-1)


def Transformer(d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300, num_decoder_layers=6, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False, return_intermediate_dec=False, group_detr=1, two_stage=False, num_feature_levels=4, dec_n_points=4, lite_refpoint_refine=False, decoder_norm_type="LN", bbox_reparam=False, name="transformer"):  # fmt: skip
    query = Input(shape=(None, d_model), name="materialize_query")
    memory = Input(shape=(None, d_model), name="materialize_memory")
    sine = Input(shape=(None, 2 * d_model), name="materialize_sine")
    layer_args = (d_model, sa_nhead, ca_nhead, dim_feedforward, dropout)
    deform_args = (num_feature_levels, dec_n_points)
    outputs = []
    for layer in range(num_decoder_layers):
        layer_name = DECODER_LAYER.format(layer)
        outputs += materialize_decoder_layer(query, memory, *layer_args, *deform_args, layer_name)  # fmt: skip
    outputs.append(mlp(sine, 2 * d_model, d_model, d_model, 2, REFERENCE_POINT_HEAD))  # fmt: skip
    outputs.append(build_layer_norm(DECODER_NORM)(query))
    if two_stage:
        outputs += materialize_encoder_outputs(memory, d_model, group_detr)
    model = Model([query, memory, sine], outputs, name=name)
    model.d_model = d_model
    keys = ("d_model", "sa_nhead", "ca_nhead", "num_queries", "num_decoder_layers", "dim_feedforward", "dropout", "activation", "normalize_before", "return_intermediate_dec", "group_detr", "two_stage", "num_feature_levels", "dec_n_points", "lite_refpoint_refine", "decoder_norm_type", "bbox_reparam")  # fmt: skip
    values = (d_model, sa_nhead, ca_nhead, num_queries, num_decoder_layers, dim_feedforward, dropout, activation, normalize_before, return_intermediate_dec, group_detr, two_stage, num_feature_levels, dec_n_points, lite_refpoint_refine, decoder_norm_type, bbox_reparam)  # fmt: skip
    model.transformer_config = dict(zip(keys, values))
    return model


def materialize_encoder_outputs(memory, d_model, group_detr):
    outputs = []
    for group in range(group_detr):
        name = ENCODER_OUTPUT.format(group)
        projected = layers.Dense(d_model, name=name)(memory)
        norm_name = ENCODER_OUTPUT_NORM.format(group)
        outputs.append(build_layer_norm(norm_name)(projected))
    return outputs


# position_embeddings is unused by this decoder-only transformer (LW-DETR
# encodes in the backbone) but stays in the signature: callers and the test
# mock mirror the upstream DETR argument order positionally.
def apply_transformer(model, sources, masks, position_embeddings, bbox_embed, enc_out_class_embed, enc_out_bbox_embed, query_feat, refpoint_embed, training):  # fmt: skip
    config = model.transformer_config
    memory, spatial_shapes = flatten_sources(sources)
    padding_mask = flatten_masks(masks)
    valid_ratios = compute_valid_ratios(masks)
    proposals = None
    if config["two_stage"]:
        args = (model, memory, padding_mask, spatial_shapes, config)
        heads = (enc_out_class_embed, enc_out_bbox_embed)
        proposals = select_encoder_proposals(*args, *heads, training)
    hidden, references = None, None
    if config["num_decoder_layers"] > 0:
        batch = ops.shape(memory)[0]
        queries = build_decoder_queries(query_feat, refpoint_embed, batch, proposals, config)  # fmt: skip
        args = (model, queries.target, memory, config, bbox_embed, padding_mask)
        tail = (spatial_shapes, valid_ratios, training)
        hidden, references = apply_decoder(*args, queries.refpoints, *tail)
    return collect_transformer_outputs(hidden, references, proposals, config)


def flatten_sources(sources):
    tokens = []
    spatial_shapes = []
    for source in sources:
        batch = ops.shape(source)[0]
        channels = ops.shape(source)[3]
        tokens.append(ops.reshape(source, (batch, -1, channels)))
        spatial_shapes.append((ops.shape(source)[1], ops.shape(source)[2]))
    return ops.concatenate(tokens, axis=1), spatial_shapes


def flatten_masks(masks):
    padding_mask = None
    if masks is not None:
        flat = [ops.reshape(m, (ops.shape(m)[0], -1)) for m in masks]
        padding_mask = ops.concatenate(flat, axis=1)
    return padding_mask


def compute_valid_ratios(masks):
    valid_ratios = None
    if masks is not None:
        valid_ratios = ops.stack([get_valid_ratio(m) for m in masks], axis=1)
    return valid_ratios


def select_encoder_proposals(model, memory, padding_mask, spatial_shapes, config, class_heads, bbox_heads, training):  # fmt: skip
    reparam = config["bbox_reparam"]
    args = (memory, padding_mask, spatial_shapes)
    encoded, proposals = gen_encoder_output_proposals(*args, not reparam)
    group_detr = config["group_detr"] if training else 1
    selections = []
    for group in range(group_detr):
        args = (model, encoded, proposals, config)
        heads = (class_heads[group], bbox_heads[group])
        selections.append(select_group_proposals(*args, *heads, group))
    joined = [ops.concatenate(field, axis=1) for field in zip(*selections)]
    return EncoderProposals(*joined)


def select_group_proposals(model, encoded, proposals, config, class_head, bbox_head, group):  # fmt: skip
    projected = model.get_layer(ENCODER_OUTPUT.format(group))(encoded)
    normalized = model.get_layer(ENCODER_OUTPUT_NORM.format(group))(projected)
    logits = class_head(normalized)
    deltas = bbox_head(normalized)
    coordinates = refine_boxes(deltas, proposals, config["bbox_reparam"])
    topk = min(config["num_queries"], ops.shape(logits)[-2])
    ranked = ops.top_k(ops.max(logits, axis=-1), topk)[1]
    indices = ops.expand_dims(ranked, axis=-1)
    boxes = ops.take_along_axis(coordinates, indices, axis=1)
    memory = ops.take_along_axis(normalized, indices, axis=1)
    return EncoderProposals(ops.stop_gradient(boxes), memory, boxes)


def build_decoder_queries(query_feat, refpoint_embed, batch, proposals, config):
    target = ops.repeat(ops.expand_dims(query_feat, axis=0), batch, axis=0)
    stacked = ops.expand_dims(refpoint_embed, axis=0)
    refpoints = ops.repeat(stacked, batch, axis=0)
    if proposals is not None:
        args = (refpoints, proposals.refpoints, config["bbox_reparam"])
        refpoints = merge_two_stage_refpoints(*args)
    return DecoderQueries(target, refpoints)


def merge_two_stage_refpoints(refpoints, selected, bbox_reparam):
    length = ops.shape(selected)[-2]
    head = refine_boxes(refpoints[..., :length, :], selected, bbox_reparam)
    tail = refpoints[..., length:, :]
    return ops.concatenate([head, tail], axis=-2)


def collect_transformer_outputs(hidden, references, proposals, config):
    memory, boxes = None, None
    if proposals is not None:
        memory = proposals.memory
        boxes = proposals.boxes
        if not config["bbox_reparam"]:
            boxes = ops.sigmoid(boxes)
    return hidden, references, memory, boxes
