"""Ports the published PF-RPN PyTorch checkpoint onto the paz model.

Layers are named after the checkpoint entry they load, with dots written as
underscores, so every layer finds its weights by name. Only the entries that
paz builds through shared helpers with their own naming are renamed first.

Needs ``torch`` to read the checkpoint, so this module is imported by the
converter and the parity test, never by the model itself.
"""
import numpy as np
import torch
from keras.layers import Conv2D, Dense, EinsumDense
from keras.layers import GroupNormalization, LayerNormalization

from paz.models.transformers.embeddings.token import LearnableTokens


def convert(model, checkpoint_path, output_path):
    port_weights(model, checkpoint_path).save_weights(output_path)


def port_weights(model, checkpoint_path):
    weights = read_checkpoint(checkpoint_path)
    ported = Dense, EinsumDense, Conv2D, LearnableTokens
    ported = ported + (LayerNormalization, GroupNormalization)
    for layer in model._flatten_layers():
        if isinstance(layer, ported):
            port_layer(layer, weights)
    return model


def read_checkpoint(checkpoint_path):
    kwargs = dict(map_location="cpu", weights_only=True)
    state = torch.load(checkpoint_path, **kwargs)
    weights = {}
    for key, value in state.items():
        weights.update(split_fused_attention(key, value.numpy()))
    return {rename(key): value for key, value in weights.items()}


def split_fused_attention(key, value):
    """Splits a fused query, key and value projection into its three parts."""
    prefix, fused, kind = key.partition(".attn.in_proj_")
    if fused:
        parts = "query", "key", "value"
        names = [f"{prefix}.{part}.{kind}" for part in parts]
        split = dict(zip(names, np.split(value, 3, axis=0)))
    else:
        split = {key: value}
    return split


def rename(key):
    """Renames the entries of layers that paz helpers name themselves."""
    names = {
        ".sampling_offsets.": ".offsets.",
        ".attention_weights.": ".weights.",
        ".value_proj.": ".values.",
        ".output_proj.": ".output.",
        ".attn.v_proj.": ".cells_query.",
        ".attn.l_proj.": ".texts_key.",
        ".attn.values_l_proj.": ".cells_values.",
        ".attn.out_v_proj.": ".cells_output.",
        ".gamma_v": ".cells_scale",
        ".attn.values_v_proj.": ".texts_values.",
        ".attn.out_l_proj.": ".texts_output.",
        ".gamma_l": ".texts_scale",
        ".q_proj.": ".query.",
        ".k_proj.": ".key.",
        ".v_proj.": ".value.",
        ".c_proj.": ".attention_output.",
        "_attn.proj.": "_attn.attention_output.",
        ".attn.out_proj.": ".attention_output.",
    }
    for old, new in names.items():
        key = key.replace(old, new)
    return key.replace(".", "_")


def port_layer(layer, weights):
    shapes = [tuple(weight.shape) for weight in layer.weights]
    kernel = weights.get(f"{layer.name}_weight", weights.get(layer.name))
    bias = weights.get(f"{layer.name}_bias", np.zeros(shapes[-1], "float32"))
    values = [orient(layer, kernel), bias][:len(shapes)]
    values = [value.reshape(shape) for value, shape in zip(values, shapes)]
    layer.set_weights(values)


def orient(layer, kernel):
    """Moves the output axis last, where Keras keeps it."""
    if kernel.ndim == 4:
        oriented = np.transpose(kernel, (2, 3, 1, 0))
    elif isinstance(layer, (Dense, EinsumDense)):
        oriented = kernel.T
    else:
        oriented = kernel
    return oriented
