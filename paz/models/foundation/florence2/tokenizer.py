"""Byte-level BPE encoder for the Florence-2 BART tokenizer.

Loads the Hugging Face ``tokenizer.json`` shipped with the pretrained
weights and reproduces ``BartTokenizer`` encoding: GPT-2 byte-level
pre-tokenization, BPE merges, then ``<s> ... </s>`` wrapping.
"""
import json
import re
from collections import namedtuple

from paz.models.transformers.tokenizers import build_byte_to_char
from paz.models.transformers.tokenizers import build_merge_ranks
from paz.models.transformers.tokenizers import merge_pairs

FLOW_TOKEN_ID = 51289
GPT2_SPLITS = (r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+"
               r"| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
SPLIT_PATTERN = re.compile(GPT2_SPLITS)
Tokenizer = namedtuple("Tokenizer", "vocabulary merge_ranks byte_to_char")


def load_tokenizer(path):
    source = json.loads(open(path).read())
    vocabulary = source["model"]["vocab"]
    merge_ranks = build_merge_ranks(source["model"]["merges"])
    return Tokenizer(vocabulary, merge_ranks, build_byte_to_char())


def encode(tokenizer, text, max_length=77):
    start_of_text, end_of_text = 0, 2
    token_ids = [start_of_text]
    for piece in SPLIT_PATTERN.findall(text):
        token_ids.extend(encode_piece(tokenizer, piece))
    token_ids = token_ids[:max_length - 1]
    token_ids.append(end_of_text)
    return token_ids


def encode_piece(tokenizer, piece):
    visible = to_visible_characters(tokenizer, piece)
    parts = merge_pairs(list(visible), tokenizer.merge_ranks)
    return [tokenizer.vocabulary[part] for part in parts]


def to_visible_characters(tokenizer, piece):
    piece_bytes = piece.encode("utf-8")
    return "".join(tokenizer.byte_to_char[byte] for byte in piece_bytes)


def build_policy_prompt(instruction):
    meta = ("Agent Type: 1-arm Franka Panda, "
            "Action Space: Delta End-Effector, ")
    return f"{meta} Task Instruction: {instruction}"
