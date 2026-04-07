import json
from pathlib import Path


def decode_whisper_tokens(token_ids, vocabulary_path=None, config_path=None):
    vocabulary_path = vocabulary_path or build_vocabulary_path()
    id_to_bytes = build_token_id_to_bytes(vocabulary_path)
    special_to_bytes = build_special_id_to_bytes(config_path)
    return decode_token_ids(token_ids, id_to_bytes, special_to_bytes)


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
        range(ord("\u00a1"), ord("\u00ac") + 1),
        range(ord("\u00ae"), ord("\u00ff") + 1),
    ]
    values = set()
    for byte_range in ranges:
        values.update(byte_range)
    return values


def build_special_id_to_bytes(config_path=None):
    token_map = build_special_token_map(config_path)
    id_to_bytes = {}
    for text, token_id in token_map.items():
        id_to_bytes[token_id] = text.encode("utf-8")
    return id_to_bytes


def find_special_token_id(token_text, config_path=None):
    token_map = build_special_token_map(config_path)
    if token_text not in token_map:
        message = "Unknown special token: {}".format(token_text)
        raise KeyError(message)
    return token_map[token_text]


def build_special_token_map(config_path=None):
    config_path = config_path or build_tokenizer_config_path()
    config = load_tokenizer_config(config_path)
    return dict(config["config"]["special_tokens"])


def load_tokenizer_config(config_path):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tokenizer_config_path():
    return Path(__file__).with_name("tokenizer_data") / "tokenizer.json"


def build_vocabulary_path():
    return Path(__file__).with_name("vocabulary.json")


if __name__ == "__main__":
    token_ids = [50257, 50357, 50362, 31373, 13, 50256]
    print(decode_whisper_tokens(token_ids))
