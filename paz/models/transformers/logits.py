"""Logit-shaping transforms.

The sampling transforms run over (batch, vocabulary): top_k <= 0 disables
top-k; top_p >= 1 disables nucleus truncation. ``soft_cap`` is an in-graph
tanh cap on attention or output logits and uses ``keras.ops`` so it composes
with symbolic Keras tensors.
"""
import jax
import jax.numpy as jp
from keras import ops


def soft_cap(values, cap):
    if cap is None:
        return values
    return ops.multiply(ops.tanh(ops.divide(values, cap)), cap)


def apply_temperature(logits, temperature):
    return logits / temperature


def apply_top_k(logits, top_k):
    if not top_k or top_k <= 0:
        return logits
    values, _ = jax.lax.top_k(logits, top_k)
    threshold = values[..., -1:]
    return jp.where(logits < threshold, -jp.inf, logits)


def apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    order = jp.argsort(-logits, axis=-1)
    sorted_logits = jp.take_along_axis(logits, order, axis=-1)
    probs = jax.nn.softmax(sorted_logits, axis=-1)
    prefix = jp.cumsum(probs, axis=-1) - probs
    sorted_logits = jp.where(prefix < top_p, sorted_logits, -jp.inf)
    inverse = jp.argsort(order, axis=-1)
    return jp.take_along_axis(sorted_logits, inverse, axis=-1)
