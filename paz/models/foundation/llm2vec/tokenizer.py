"""Llama-3 byte-level BPE tokenizer for the LLM2Vec text encoder.

Reads the Hugging Face ``tokenizer.json`` shipped with the weights: added
special tokens such as ``<|eot_id|>`` are matched verbatim, the text in
between is split with the Llama-3 pre-tokenizer pattern, mapped to the
GPT-2 visible byte alphabet and merged by rank.
"""
import json
from collections import namedtuple
from pathlib import Path

import regex

from paz.models.transformers import tokenizers as bpe

BEGIN_OF_TEXT = "<|begin_of_text|>"
Tokenizer = namedtuple(
    "Tokenizer",
    "vocabulary merge_ranks byte_to_char special_ids special_pattern "
    "split_pattern",
)


def load_tokenizer(path):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    vocabulary = dict(source["model"]["vocab"])
    merge_ranks = bpe.build_merge_ranks(source["model"]["merges"])
    special_ids = {}
    for token in source["added_tokens"]:
        special_ids[token["content"]] = token["id"]
    split_pattern = regex.compile(read_split_pattern(source))
    args = (vocabulary, merge_ranks, bpe.build_byte_to_char(), special_ids,
            build_special_pattern(special_ids), split_pattern)
    return Tokenizer(*args)


def read_split_pattern(source):
    splitters = source["pre_tokenizer"]["pretokenizers"]
    return splitters[0]["pattern"]["Regex"]


def build_special_pattern(special_ids):
    longest_first = sorted(special_ids, key=len, reverse=True)
    return regex.compile("|".join(regex.escape(t) for t in longest_first))


def encode(tokenizer, text):
    """Token ids of ``text``; special tokens pass through, no BOS added."""
    token_ids = []
    cursor = 0
    for match in tokenizer.special_pattern.finditer(text):
        token_ids.extend(encode_plain(tokenizer, text[cursor:match.start()]))
        token_ids.append(tokenizer.special_ids[match.group()])
        cursor = match.end()
    token_ids.extend(encode_plain(tokenizer, text[cursor:]))
    return token_ids


def encode_plain(tokenizer, text):
    token_ids = []
    for piece in tokenizer.split_pattern.findall(text):
        visible = bpe.to_visible_characters(tokenizer.byte_to_char, piece)
        parts = bpe.merge_pairs(list(visible), tokenizer.merge_ranks)
        token_ids.extend(tokenizer.vocabulary[part] for part in parts)
    return token_ids
