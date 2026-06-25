"""Token sampling for Gemma4 generation: temperature, top-k, top-p.

Pure functions over logits shaped (batch, vocabulary). `sample_logits` composes
temperature scaling, top-k truncation and top-p (nucleus) truncation, then draws
one token per row with `jax.random.categorical`. Greedy decoding is NOT done
here: it is plain `argmax`, which breaks ties deterministically by lowest index;
`categorical` would instead break the frequent bfloat16 ties stochastically.
"""
from collections import namedtuple

import jax

from paz.models.transformers.logits import apply_temperature
from paz.models.transformers.logits import apply_top_k
from paz.models.transformers.logits import apply_top_p

# top_k <= 0 disables top-k; top_p >= 1 disables nucleus truncation.
SamplingArgs = namedtuple("SamplingArgs", "temperature top_k top_p")


def sample_logits(logits, key, args):
    logits = apply_temperature(logits, args.temperature)
    logits = apply_top_k(logits, args.top_k)
    logits = apply_top_p(logits, args.top_p)
    return jax.random.categorical(key, logits, axis=-1).astype("int32")
