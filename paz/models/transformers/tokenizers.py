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
