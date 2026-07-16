"""Byte-level BPE encoder for the Florence-2 BART tokenizer.

Loads the Hugging Face ``tokenizer.json`` shipped with the pretrained
weights and reproduces ``BartTokenizer`` encoding: GPT-2 byte-level
pre-tokenization, BPE merges, then ``<s> ... </s>`` wrapping.
"""
import json
import re
from collections import namedtuple

from paz.models.transformers.tokenizers import build_character_to_byte

BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 1
FLOW_TOKEN_ID = 51289
SPLIT_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+"
    r"| ?[^\s\w]+|\s+(?!\S)|\s+"
)
Tokenizer = namedtuple("Tokenizer", "vocabulary merge_ranks byte_to_char")


def load_tokenizer(path):
    source = json.loads(open(path).read())
    vocabulary = source["model"]["vocab"]
    merges = source["model"]["merges"]
    pairs = [tuple(merge.split(" ")) for merge in merges]
    merge_ranks = {pair: rank for rank, pair in enumerate(pairs)}
    byte_to_char = build_byte_to_char()
    return Tokenizer(vocabulary, merge_ranks, byte_to_char)


def build_byte_to_char():
    character_to_byte = build_character_to_byte()
    return {byte: char for char, byte in character_to_byte.items()}


def encode(tokenizer, text, max_length=77):
    token_ids = [BOS_TOKEN_ID]
    for piece in SPLIT_PATTERN.findall(text):
        token_ids.extend(encode_piece(tokenizer, piece))
    token_ids = token_ids[:max_length - 1]
    token_ids.append(EOS_TOKEN_ID)
    return token_ids


def encode_piece(tokenizer, piece):
    visible = to_visible_characters(tokenizer, piece)
    parts = merge_pairs(list(visible), tokenizer.merge_ranks)
    return [tokenizer.vocabulary[part] for part in parts]


def to_visible_characters(tokenizer, piece):
    piece_bytes = piece.encode("utf-8")
    return "".join(tokenizer.byte_to_char[byte] for byte in piece_bytes)


def merge_pairs(parts, merge_ranks):
    while len(parts) > 1:
        pair = find_best_pair(parts, merge_ranks)
        if pair is None:
            break
        parts = merge_pair(parts, pair)
    return parts


def find_best_pair(parts, merge_ranks):
    best_pair, best_rank = None, None
    for pair in zip(parts[:-1], parts[1:]):
        rank = merge_ranks.get(pair)
        if rank is None:
            continue
        if best_rank is None or rank < best_rank:
            best_pair, best_rank = pair, rank
    return best_pair


def merge_pair(parts, pair):
    merged, index = [], 0
    while index < len(parts):
        is_pair_start = (
            index < len(parts) - 1
            and (parts[index], parts[index + 1]) == pair)
        if is_pair_start:
            merged.append(parts[index] + parts[index + 1])
            index = index + 2
        else:
            merged.append(parts[index])
            index = index + 1
    return merged


def build_policy_prompt(instruction):
    meta = ("Agent Type: 1-arm Franka Panda, "
            "Action Space: Delta End-Effector, ")
    return f"{meta} Task Instruction: {instruction}"
