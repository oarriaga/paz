"""Ports published RF-DETR PyTorch checkpoints onto the paz models.

Only the first query group is ported: group-DETR duplicates the query tables
and the first-stage heads during training, and the reference uses group zero
at inference.

Needs ``torch`` to read the checkpoint, so this module is imported by the
converter and the parity test, never by the models themselves.
"""
import numpy as np
import torch

from paz.models.detection.rf_detr.models import NUM_QUERIES
from paz.models.detection.rf_detr.models import PROJECTOR_BLOCKS

BACKBONE = "backbone.0.encoder.encoder"
PROJECTOR = "backbone.0.projector.stages.0"
DECODER = "transformer.decoder"


def port_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    weights = {key: value.numpy() for key, value in checkpoint["model"].items()}
    port_backbone(model, weights, read_grid(model))
    port_projector(model, weights, PROJECTOR_BLOCKS)
    port_decoder(model, weights, NUM_QUERIES)
    return model


def convert(model, checkpoint_path, output_path):
    port_weights(model, checkpoint_path).save_weights(output_path)


def port_backbone(model, weights, grid):
    port_patch_embedding(model, weights, grid)
    # Blocks past the last tapped one feed nothing, so Keras prunes them out
    # of the graph and their checkpoint weights stay unused.
    for block in range(count_blocks(model)):
        port_block(model, weights, block)
    set_norm(model, "norm", weights, f"{BACKBONE}.layernorm")


def count_blocks(model):
    names = {layer.name for layer in model.layers}
    count = 0
    while f"block_{count}_norm1" in names:
        count = count + 1
    return count


def port_patch_embedding(model, weights, grid):
    key = f"{BACKBONE}.embeddings.patch_embeddings.projection"
    kernel = np.transpose(weights[f"{key}.weight"], (2, 3, 1, 0))
    layer = model.get_layer("patch_embed_proj")
    layer.set_weights([kernel, weights[f"{key}.bias"]])
    class_token = weights[f"{BACKBONE}.embeddings.cls_token"].reshape(1, -1)
    model.get_layer("cls_token").set_weights([class_token])
    table = weights[f"{BACKBONE}.embeddings.position_embeddings"][0]
    model.get_layer("pos_embed").set_weights([resize_positions(table, grid)])


def resize_positions(table, grid):
    """Resamples the stored position grid onto this model's patch grid.

    Published checkpoints keep the grid the backbone was pretrained on and the
    reference interpolates it on every forward pass. Doing it once here keeps
    the graph free of it and gives the same numbers.
    """
    side = int(round((table.shape[0] - 1) ** 0.5))
    if (side, side) == grid:
        resized = table
    else:
        patches = table[1:].reshape(1, side, side, -1)
        patches = np.transpose(patches, (0, 3, 1, 2))
        resized = interpolate_bicubic(patches, grid)
        resized = np.concatenate([table[:1], resized], axis=0)
    return resized


def interpolate_bicubic(patches, grid):
    kwargs = dict(mode="bicubic", align_corners=False, antialias=True)
    tensor = torch.from_numpy(np.ascontiguousarray(patches))
    resized = torch.nn.functional.interpolate(tensor, size=grid, **kwargs)
    resized = resized.permute(0, 2, 3, 1).numpy()
    return resized.reshape(grid[0] * grid[1], -1)


def port_block(model, weights, block):
    key = f"{BACKBONE}.encoder.layer.{block}"
    name = f"block_{block}"
    set_norm(model, f"{name}_norm1", weights, f"{key}.norm1")
    port_fused_attention(model, weights, f"{key}.attention", name)
    set_scale(model, f"{name}_ls1", weights, f"{key}.layer_scale1")
    set_norm(model, f"{name}_norm2", weights, f"{key}.norm2")
    set_dense(model, f"{name}_mlp_fc1", weights, f"{key}.mlp.fc1")
    set_dense(model, f"{name}_mlp_fc2", weights, f"{key}.mlp.fc2")
    set_scale(model, f"{name}_ls2", weights, f"{key}.layer_scale2")


def port_fused_attention(model, weights, key, name):
    parts = [f"{key}.attention.{part}" for part in ("query", "key", "value")]
    kernels = [weights[f"{part}.weight"].T for part in parts]
    biases = [weights[f"{part}.bias"] for part in parts]
    fused = [np.concatenate(kernels, axis=1), np.concatenate(biases, axis=0)]
    model.get_layer(f"{name}_qkv").set_weights(fused)
    set_dense(model, f"{name}_proj", weights, f"{key}.output.dense")


def port_projector(model, weights, num_blocks):
    port_convolution(model, weights, "projector_cv1", f"{PROJECTOR}.0.cv1")
    for index in range(num_blocks):
        key = f"{PROJECTOR}.0.m.{index}"
        name = f"projector_m_{index}"
        port_convolution(model, weights, f"{name}_cv1", f"{key}.cv1")
        port_convolution(model, weights, f"{name}_cv2", f"{key}.cv2")
    port_convolution(model, weights, "projector_cv2", f"{PROJECTOR}.0.cv2")
    set_norm(model, "projector_norm", weights, f"{PROJECTOR}.1")


def port_convolution(model, weights, name, key):
    kernel = np.transpose(weights[f"{key}.conv.weight"], (2, 3, 1, 0))
    model.get_layer(f"{name}_conv").set_weights([kernel])
    set_norm(model, f"{name}_norm", weights, f"{key}.bn")


def port_decoder(model, weights, num_queries):
    port_first_stage(model, weights, num_queries)
    for index in (0, 1):
        key = f"{DECODER}.ref_point_head.layers.{index}"
        set_dense(model, f"decoder_ref_point_head_{index}", weights, key)
    for layer in range(count_prefixed(weights, f"{DECODER}.layers")):
        port_decoder_layer(model, weights, layer)
    set_norm(model, "decoder_norm", weights, f"{DECODER}.norm")
    set_dense(model, "class_embed", weights, "class_embed")
    port_box_head(model, weights, "bbox_embed", "bbox_embed")


def port_first_stage(model, weights, num_queries):
    set_dense(model, "enc_output", weights, "transformer.enc_output.0")
    key = "transformer.enc_output_norm.0"
    set_norm(model, "enc_output_norm", weights, key)
    key = "transformer.enc_out_class_embed.0"
    set_dense(model, "enc_class_embed", weights, key)
    key = "transformer.enc_out_bbox_embed.0"
    port_box_head(model, weights, "enc_bbox_embed", key)
    port_table(model, weights, "refpoint_embed", num_queries)
    port_table(model, weights, "query_feat", num_queries)


def port_decoder_layer(model, weights, layer):
    key = f"{DECODER}.layers.{layer}"
    name = f"decoder_{layer}"
    port_self_attention(model, weights, f"{key}.self_attn", name)
    port_cross_attention(model, weights, f"{key}.cross_attn", name)
    set_dense(model, f"{name}_linear1", weights, f"{key}.linear1")
    set_dense(model, f"{name}_linear2", weights, f"{key}.linear2")
    for index in (1, 2, 3):
        set_norm(model, f"{name}_norm{index}", weights, f"{key}.norm{index}")


def port_self_attention(model, weights, key, name):
    kernels = np.split(weights[f"{key}.in_proj_weight"], 3, axis=0)
    biases = np.split(weights[f"{key}.in_proj_bias"], 3, axis=0)
    parts = ("query", "key", "value")
    for part, kernel, bias in zip(parts, kernels, biases):
        layer = model.get_layer(f"{name}_{part}")
        shape = layer.kernel.shape
        layer.set_weights([kernel.T.reshape(shape), bias.reshape(shape[1:])])
    layer = model.get_layer(f"{name}_attention_output")
    kernel = weights[f"{key}.out_proj.weight"].T.reshape(layer.kernel.shape)
    layer.set_weights([kernel, weights[f"{key}.out_proj.bias"]])


def port_cross_attention(model, weights, key, name):
    parts = ("offsets", "sampling_offsets"), ("weights", "attention_weights")
    parts = parts + (("values", "value_proj"), ("output", "output_proj"))
    for suffix, part in parts:
        set_dense(model, f"{name}_cross_{suffix}", weights, f"{key}.{part}")


def port_box_head(model, weights, name, key):
    for index in range(3):
        set_dense(model, f"{name}_{index}", weights, f"{key}.layers.{index}")


def port_table(model, weights, name, num_queries):
    model.get_layer(name).set_weights([weights[f"{name}.weight"][:num_queries]])


def set_dense(model, name, weights, key):
    bias = weights[f"{key}.bias"]
    model.get_layer(name).set_weights([weights[f"{key}.weight"].T, bias])


def set_norm(model, name, weights, key):
    bias = weights[f"{key}.bias"]
    model.get_layer(name).set_weights([weights[f"{key}.weight"], bias])


def set_scale(model, name, weights, key):
    model.get_layer(name).set_weights([weights[f"{key}.lambda1"]])


def count_prefixed(weights, prefix):
    indices = set()
    for key in weights:
        if key.startswith(f"{prefix}."):
            indices.add(int(key[len(prefix) + 1:].split(".")[0]))
    return len(indices)


def read_grid(model):
    patch_size = model.get_layer("patch_embed_proj").kernel_size[0]
    height, width = model.input_shape[1], model.input_shape[2]
    return height // patch_size, width // patch_size
