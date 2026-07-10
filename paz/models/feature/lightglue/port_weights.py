from collections import namedtuple

import jax.numpy as jp
import torch

Linear = namedtuple("Linear", ["weight", "bias"])
FeedForward = namedtuple("FeedForward", ["input", "norm", "output"])
SelfAttention = namedtuple("SelfAttention", ["qkv", "out_proj", "ffn"])
CrossAttention = namedtuple("CrossAttention", ["qk", "v", "out_proj", "ffn"])
Layer = namedtuple("Layer", ["self_attn", "cross_attn"])
Assignment = namedtuple("Assignment", ["final_proj", "matchability"])
LightGlueParams = namedtuple(
    "LightGlueParams", ["posenc", "input_proj", "layers", "assign"])


def port_weights(torch_path, num_layers=6):
    state = matcher_state(torch.load(torch_path, map_location="cpu"))
    layers = [build_layer(state, index) for index in range(num_layers)]
    return LightGlueParams(
        posenc=array(state["posenc.Wr.weight"]).T,
        input_proj=linear(state, "input_proj"),
        layers=layers,
        assign=build_assignment(state, num_layers - 1))


def matcher_state(state):
    return {key[len("matcher."):]: value for key, value in state.items()
            if key.startswith("matcher.")}


def build_layer(state, index):
    prefix = f"transformers.{index}"
    self_attn = SelfAttention(
        qkv=linear(state, f"{prefix}.self_attn.Wqkv"),
        out_proj=linear(state, f"{prefix}.self_attn.out_proj"),
        ffn=feed_forward(state, f"{prefix}.self_attn.ffn"))
    cross_attn = CrossAttention(
        qk=linear(state, f"{prefix}.cross_attn.to_qk"),
        v=linear(state, f"{prefix}.cross_attn.to_v"),
        out_proj=linear(state, f"{prefix}.cross_attn.to_out"),
        ffn=feed_forward(state, f"{prefix}.cross_attn.ffn"))
    return Layer(self_attn, cross_attn)


def build_assignment(state, index):
    prefix = f"log_assignment.{index}"
    return Assignment(final_proj=linear(state, f"{prefix}.final_proj"),
                      matchability=linear(state, f"{prefix}.matchability"))


def feed_forward(state, prefix):
    return FeedForward(input=linear(state, f"{prefix}.0"),
                       norm=norm(state, f"{prefix}.1"),
                       output=linear(state, f"{prefix}.3"))


def linear(state, prefix):
    weight = array(state[f"{prefix}.weight"]).T
    return Linear(weight, array(state[f"{prefix}.bias"]))


def norm(state, prefix):
    return Linear(array(state[f"{prefix}.weight"]),
                  array(state[f"{prefix}.bias"]))


def array(tensor):
    return jp.asarray(tensor.numpy(), jp.float32)
