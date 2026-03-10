import json
from pathlib import Path


def build_whisper_base_en_vocabulary_path():
    return Path(__file__).with_name("vocabulary.json")


def build_whisper_base_en_tokenizer_config_path():
    return Path(__file__).with_name("tokenizer_data") / "tokenizer.json"


def load_tokenizer_config(tokenizer_config_path=None):
    if tokenizer_config_path is None:
        tokenizer_config_path = build_whisper_base_en_tokenizer_config_path()
    tokenizer_config_path = Path(tokenizer_config_path)
    with open(tokenizer_config_path, "r", encoding="utf-8") as filedata:
        return json.load(filedata)


def build_special_token_text_to_id(tokenizer_config_path=None):
    tokenizer_config = load_tokenizer_config(tokenizer_config_path)
    return dict(tokenizer_config["config"]["special_tokens"])


def find_special_token_id(token_text, tokenizer_config_path=None):
    special_token_text_to_id = build_special_token_text_to_id(
        tokenizer_config_path
    )
    if token_text not in special_token_text_to_id:
        raise KeyError("Unknown special token: {}".format(token_text))
    return special_token_text_to_id[token_text]


def build_character_to_byte():
    visible_byte_values = set(
        [
            *range(ord("!"), ord("~") + 1),
            *range(ord("¡"), ord("¬") + 1),
            *range(ord("®"), ord("ÿ") + 1),
        ]
    )
    character_to_byte = {}
    next_character_number = 256
    for byte_value in range(256):
        if byte_value in visible_byte_values:
            character = chr(byte_value)
        else:
            character = chr(next_character_number)
            next_character_number = next_character_number + 1
        character_to_byte[character] = byte_value
    return character_to_byte


def build_special_token_id_to_bytes(tokenizer_config_path=None):
    special_token_text_to_id = build_special_token_text_to_id(
        tokenizer_config_path
    )
    token_id_to_bytes = {}
    for token_text, token_id in special_token_text_to_id.items():
        token_id_to_bytes[token_id] = token_text.encode("utf-8")
    return token_id_to_bytes


def build_token_id_to_bytes(vocabulary_filepath=None):

    def text_to_bytes(token_text, character_to_byte):
        return bytes(character_to_byte[character] for character in token_text)

    if vocabulary_filepath is None:
        vocabulary_filepath = build_whisper_base_en_vocabulary_path()
    vocabulary_filepath = Path(vocabulary_filepath)
    with open(vocabulary_filepath, "r", encoding="utf-8") as filedata:
        token_text_to_id = json.load(filedata)
    character_to_byte = build_character_to_byte()
    token_id_to_bytes = {}
    for token_text, token_id in token_text_to_id.items():
        token_id_to_bytes[token_id] = text_to_bytes(token_text, character_to_byte)
    return token_id_to_bytes


def filter_special_token_ids(token_ids, tokenizer_config_path=None):
    special_token_text_to_id = build_special_token_text_to_id(
        tokenizer_config_path
    )
    special_token_ids = set(special_token_text_to_id.values())
    return [token_id for token_id in token_ids if token_id not in special_token_ids]


def decode_token_ids(token_ids, token_id_to_bytes, special_token_id_to_bytes):
    output = bytearray()
    for token_id in token_ids:
        if token_id in token_id_to_bytes:
            token_bytes = token_id_to_bytes[token_id]
        elif token_id in special_token_id_to_bytes:
            token_bytes = special_token_id_to_bytes[token_id]
        else:
            raise KeyError("Unknown token id: {}".format(token_id))
        output.extend(token_bytes)
    return bytes(output).decode("utf-8", errors="replace")


def decode_whisper_token_ids(
    token_ids,
    vocabulary_filepath=None,
    tokenizer_config_path=None,
    skip_special_tokens=False,
):
    if skip_special_tokens:
        token_ids = filter_special_token_ids(token_ids, tokenizer_config_path)
    token_id_to_bytes = build_token_id_to_bytes(vocabulary_filepath)
    special_token_id_to_bytes = build_special_token_id_to_bytes(
        tokenizer_config_path
    )
    return decode_token_ids(token_ids, token_id_to_bytes, special_token_id_to_bytes)


if __name__ == "__main__":
    token_ids = [50257, 50357, 50362, 31373, 13, 50256]
    print(decode_whisper_token_ids(token_ids))
