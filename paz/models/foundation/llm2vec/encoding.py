"""Prompt embedding with the LLM2Vec contract Kimodo was trained on.

A prompt is wrapped in the Llama-3 user header, prefixed with
``<|begin_of_text|>`` and closed with ``<|eot_id|>``. The final hidden
states of the prompt tokens (including ``<|eot_id|>``) are mean pooled,
one prompt at a time, as the reference encoder does.
"""
import jax
import jax.numpy as jp

import paz
from paz.models.foundation.llm2vec import tokenizer as llama_tokenizer

MAX_LENGTH = 512


def build_prompt_tokens(tokenizer, text):
    """Token ids of the wrapped prompt and how many of them are pooled."""
    header = "<|start_header_id|>user<|end_header_id|>\n\n"
    # The reference strips after prefixing a separator, so only the
    # trailing whitespace of the prompt goes.
    body = text.rstrip() + "<|eot_id|>"
    begin = tokenizer.special_ids[llama_tokenizer.BEGIN_OF_TEXT]
    token_ids = [begin] + llama_tokenizer.encode(tokenizer, header + body)
    num_pooled = len(llama_tokenizer.encode(tokenizer, body))
    return token_ids[:MAX_LENGTH], min(num_pooled, MAX_LENGTH)


def embed(model, variables, tokenizer, text):
    """Pooled ``(4096,)`` float32 embedding of one prompt."""
    token_ids, num_pooled = build_prompt_tokens(tokenizer, text)
    hidden = forward(model, variables, jp.asarray([token_ids], "int32"))
    return pool(hidden[0], num_pooled)


def pool(hidden, num_pooled):
    # The reference averages in bfloat16 before casting to float32.
    pooled = jp.mean(hidden[-num_pooled:].astype("float32"), axis=0)
    return pooled.astype(hidden.dtype).astype("float32")


def _forward(model, variables, token_ids):
    return paz.call_stateless(model, variables, token_ids)


forward = jax.jit(_forward, static_argnums=0)
