import json
from pathlib import Path

from keras.utils import get_file

from paz.models.transformers.tokenizers import build_token_id_to_bytes
from paz.models.transformers.tokenizers import decode_token_ids

ASSETS_DIR = Path(__file__).resolve().with_name("assets")
WHISPER_TOKENIZER_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.23/"  # fmt: skip


def decode_whisper_tokens(token_ids, vocabulary_path=None, config_path=None):
    vocabulary_path = vocabulary_path or resolve_vocabulary_path()
    id_to_bytes = build_token_id_to_bytes(vocabulary_path)
    special_to_bytes = build_special_id_to_bytes(config_path)
    return decode_token_ids(token_ids, id_to_bytes, special_to_bytes)


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
    return resolve_asset("tokenizer.json", "whisper_tokenizer.json")


def resolve_vocabulary_path():
    return resolve_asset("vocabulary.json", "whisper_vocabulary.json")


def resolve_asset(filename, asset_name):
    local = ASSETS_DIR / filename
    if local.exists():
        return local
    url = WHISPER_TOKENIZER_URL + asset_name
    return Path(get_file(asset_name, url, cache_subdir="paz/models/whisper"))
