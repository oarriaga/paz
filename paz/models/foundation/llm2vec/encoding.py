"""Prompt embedding with the LLM2Vec contract Kimodo was trained on.

A prompt is wrapped in the Llama-3 user header, prefixed with
``<|begin_of_text|>`` and closed with ``<|eot_id|>``. The final hidden
states of the prompt tokens (including ``<|eot_id|>``) are mean pooled,
one prompt at a time, as the reference encoder does.
"""
from functools import partial

import jax
import jax.numpy as jp

import paz
from paz.models.foundation.llm2vec import tokenizer as llama_tokenizer


def build_prompt_tokens(tokenizer, text, max_tokens=400):
    """Token ids of the wrapped prompt and how many of them are pooled."""
    header = "<|start_header_id|>user<|end_header_id|>\n\n"
    # The reference strips after prefixing a separator, so only the
    # trailing whitespace of the prompt goes.
    text = trim_prompt(tokenizer, text, max_tokens).rstrip()
    body = text + "<|eot_id|>"
    begin = tokenizer.special_ids[llama_tokenizer.BEGIN_OF_TEXT]
    token_ids = [begin] + llama_tokenizer.encode(tokenizer, header + body)
    num_pooled = len(llama_tokenizer.encode(tokenizer, body))
    return token_ids, num_pooled


def trim_prompt(tokenizer, text, max_tokens):
    """Drops whole words until the prompt fits, as the reference does."""
    num_tokens = len(llama_tokenizer.encode(tokenizer, text))
    while num_tokens > max_tokens:
        words = text.split()
        text = " ".join(words[:int(len(words) * max_tokens / num_tokens)])
        num_tokens = len(llama_tokenizer.encode(tokenizer, text))
    return text


def build_forward(model):
    """Jitted stateless forward owned by the caller, so dropping it frees
    the compiled executable together with the weights."""
    return jax.jit(partial(paz.call_stateless, model))


def embed(forward, variables, tokenizer, text):
    """Pooled ``(4096,)`` float32 embedding of one prompt."""
    token_ids, num_pooled = build_prompt_tokens(tokenizer, text)
    hidden = forward(variables, jp.asarray([token_ids], "int32"))
    return pool(hidden[0], num_pooled)


def pool(hidden, num_pooled):
    # The reference averages in bfloat16 before casting to float32.
    pooled = jp.mean(hidden[-num_pooled:].astype("float32"), axis=0)
    return pooled.astype(hidden.dtype).astype("float32")
