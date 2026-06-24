"""Token sampling for Gemma4 generation: temperature, top-k, top-p.

Pure functions over logits shaped (batch, vocabulary). `sample_logits` composes
temperature scaling, top-k truncation and top-p (nucleus) truncation, then draws
one token per row with `jax.random.categorical`. Greedy decoding is NOT done
here: it is plain `argmax`, which breaks ties deterministically by lowest index;
`categorical` would instead break the frequent bfloat16 ties stochastically.
"""
from collections import namedtuple

import jax
import jax.numpy as jnp

# top_k <= 0 disables top-k; top_p >= 1 disables nucleus truncation.
SamplingArgs = namedtuple("SamplingArgs", "temperature top_k top_p")


def sample_logits(logits, key, args):
    logits = apply_temperature(logits, args.temperature)
    logits = apply_top_k(logits, args.top_k)
    logits = apply_top_p(logits, args.top_p)
    return jax.random.categorical(key, logits, axis=-1).astype("int32")


def apply_temperature(logits, temperature):
    return logits / temperature


def apply_top_k(logits, top_k):
    if not top_k or top_k <= 0:
        return logits
    values, _ = jax.lax.top_k(logits, top_k)
    threshold = values[..., -1:]
    return jnp.where(logits < threshold, -jnp.inf, logits)


def apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    order = jnp.argsort(-logits, axis=-1)
    sorted_logits = jnp.take_along_axis(logits, order, axis=-1)
    probs = jax.nn.softmax(sorted_logits, axis=-1)
    prefix = jnp.cumsum(probs, axis=-1) - probs
    sorted_logits = jnp.where(prefix < top_p, sorted_logits, -jnp.inf)
    inverse = jnp.argsort(order, axis=-1)
    return jnp.take_along_axis(sorted_logits, inverse, axis=-1)
