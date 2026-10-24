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

HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
END_OF_TURN = "<|eot_id|>"


def build_prompt_tokens(tokenizer, text):
    """Token ids of the wrapped prompt and how many of them are pooled."""
    begin = tokenizer.special_ids[llama_tokenizer.BEGIN_OF_TEXT]
    body = text.strip() + END_OF_TURN
    token_ids = [begin] + llama_tokenizer.encode(tokenizer, HEADER + body)
    num_pooled = len(llama_tokenizer.encode(tokenizer, body))
    return token_ids, num_pooled


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
