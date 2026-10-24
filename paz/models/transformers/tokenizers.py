import json
from pathlib import Path


def decode_token_ids(token_ids, id_to_bytes, special_to_bytes):
    output = bytearray()
    for token_id in token_ids:
        if token_id in id_to_bytes:
            token_bytes = id_to_bytes[token_id]
        elif token_id in special_to_bytes:
            token_bytes = special_to_bytes[token_id]
        else:
            message = "Unknown token id: {}".format(token_id)
            raise KeyError(message)
        output.extend(token_bytes)
    return bytes(output).decode("utf-8", errors="replace")


def build_token_id_to_bytes(vocabulary_path):
    vocabulary_path = Path(vocabulary_path)
    with open(vocabulary_path, "r", encoding="utf-8") as f:
        text_to_id = json.load(f)
    char_to_byte = build_character_to_byte()
    id_to_bytes = {}
    for text, token_id in text_to_id.items():
        id_to_bytes[token_id] = text_to_bytes(text, char_to_byte)
    return id_to_bytes


def text_to_bytes(token_text, char_to_byte):
    return bytes(char_to_byte[char] for char in token_text)


def build_character_to_byte():
    visible = build_visible_byte_values()
    character_to_byte = {}
    next_number = 256
    for byte_value in range(256):
        if byte_value in visible:
            character = chr(byte_value)
        else:
            character = chr(next_number)
            next_number = next_number + 1
        character_to_byte[character] = byte_value
    return character_to_byte


def build_visible_byte_values():
    ranges = [
        range(ord("!"), ord("~") + 1),
        range(ord("¡"), ord("¬") + 1),
        range(ord("®"), ord("ÿ") + 1),
    ]
    values = set()
    for byte_range in ranges:
        values.update(byte_range)
    return values


def build_byte_to_char():
    character_to_byte = build_character_to_byte()
    return {byte: char for char, byte in character_to_byte.items()}


def to_visible_characters(byte_to_char, text):
    return "".join(byte_to_char[byte] for byte in text.encode("utf-8"))


def build_merge_ranks(merges):
    pairs = [unpack_merge(merge) for merge in merges]
    return {pair: rank for rank, pair in enumerate(pairs)}


def unpack_merge(merge):
    if isinstance(merge, str):
        return tuple(merge.split(" "))
    return tuple(merge)


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
        has_next = index < len(parts) - 1
        is_pair_start = has_next and (parts[index], parts[index + 1]) == pair
        if is_pair_start:
            merged.append(parts[index] + parts[index + 1])
            index = index + 2
        else:
            merged.append(parts[index])
            index = index + 1
    return merged
