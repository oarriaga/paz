import json
from collections import namedtuple
from pathlib import Path
import re

PIECE_NORMAL = 1
PIECE_CONTROL = 3
PIECE_USER_DEFINED = 4
PIECE_BYTE = 6

WHITESPACE = "\u2581"
SPACE_PATTERN = re.compile(r" +")
BYTE_PATTERN = re.compile(r"<0x[0-9A-Fa-f]{2}>")
IMAGE_TOKEN_TEXTS = ("<|image>", "<|image|>", "<image|>")
AUDIO_TOKEN_TEXTS = ("<|audio>", "<|audio|>", "<audio|>")
TURN_TOKEN_TEXTS = ("<|turn>", "<turn|>")
PieceArgs = namedtuple("PieceArgs", "text type index")
NormalizerArgs = namedtuple(
    "NormalizerArgs",
    "name add_dummy_prefix remove_extra_whitespaces escape_whitespaces",
)
TokenizerSource = namedtuple(
    "TokenizerSource", "normalizer denormalizer pieces merge_ranks"
)
SpecialTokenIds = namedtuple(
    "SpecialTokenIds",
    "start_token_id end_token_id pad_token_id "
    "start_of_image_token_id image_placeholder_id end_of_image_token_id "
    "start_of_audio_token_id audio_placeholder_id end_of_audio_token_id "
    "start_of_turn_token_id end_of_turn_token_id",
)


class Gemma4Tokenizer:
    def __init__(
        self,
        path,
        add_bos=False,
        add_eos=False,
        has_vision_tokens=True,
        has_audio_tokens=False,
    ):
        source = load_tokenizer_source(path)
        self.path = path
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.normalizer = source.normalizer
        self.denormalizer = source.denormalizer
        self.pieces = source.pieces
        self.merge_ranks = source.merge_ranks
        self.id_to_piece = self.pieces
        self.piece_to_id = build_piece_to_id(self.pieces)
        self.user_tree = build_prefix_tree(self.pieces, PIECE_USER_DEFINED)
        ids = build_special_token_ids(
            self.piece_to_id, has_vision_tokens, has_audio_tokens
        )
        self.special_ids = ids
        for name in ids._fields:
            setattr(self, name, getattr(ids, name))

    def vocabulary_size(self):
        return len(self.pieces)

    def get_vocabulary(self):
        return [piece.text for piece in self.pieces]

    def id_to_token(self, token_id):
        size = self.vocabulary_size()
        if token_id < 0 or token_id >= size:
            message = "`id` must be in range [0, {}].".format(size - 1)
            raise ValueError(message)
        return self.id_to_piece[token_id].text

    def token_to_id(self, token_text):
        return self.piece_to_id[token_text]

    def __call__(self, inputs):
        return self.tokenize(inputs)

    def tokenize(self, inputs):
        if isinstance(inputs, str):
            return self._tokenize_text(inputs)
        return [self._tokenize_text(text) for text in inputs]

    def detokenize(self, inputs):
        if not inputs:
            return ""
        if isinstance(inputs[0], int):
            return self._detokenize_ids(inputs)
        return [self._detokenize_ids(token_ids) for token_ids in inputs]

    def format_generation_prompt(self, prompt):
        return "".join(build_prompt_parts(prompt))

    def tokenize_generation_prompt(self, prompt):
        return self.tokenize(self.format_generation_prompt(prompt))

    def get_stop_token_ids(self):
        stop_ids = [self.end_token_id]
        if self.end_of_turn_token_id >= 0:
            stop_ids.append(self.end_of_turn_token_id)
        return tuple(stop_ids)

    def _tokenize_text(self, text):
        text = normalize_text(text, self.normalizer)
        parts = self._split_user_segments(text)
        token_ids = []
        for is_user, part in parts:
            if is_user:
                token_ids.append(self.piece_to_id[part])
                continue
            token_ids.extend(self._encode_bpe_segment(part))
        return self._add_boundary_tokens(token_ids)

    def _detokenize_ids(self, token_ids):
        pieces = [self.id_to_piece[token_id] for token_id in token_ids]
        byte_buffer = bytearray()
        output = bytearray()
        for piece in pieces:
            if piece.type == PIECE_CONTROL:
                continue
            if piece.type == PIECE_BYTE:
                byte_buffer.append(int(piece.text[3:5], 16))
                continue
            output.extend(self._flush_byte_buffer(byte_buffer))
            text = piece.text.replace(WHITESPACE, " ")
            output.extend(text.encode("utf-8"))
        output.extend(self._flush_byte_buffer(byte_buffer))
        text = output.decode("utf-8", errors="replace")
        return denormalize_text(text, self.denormalizer)

    def _add_boundary_tokens(self, token_ids):
        output = list(token_ids)
        if self.add_bos:
            output.insert(0, self.start_token_id)
        if self.add_eos:
            output.append(self.end_token_id)
        return output

    def _split_user_segments(self, text):
        parts = []
        start = 0
        index = 0
        while index < len(text):
            token_text = self._find_longest_match(text, index)
            if token_text is None:
                index = index + 1
                continue
            if start != index:
                parts.append((False, text[start:index]))
            parts.append((True, token_text))
            index = index + len(token_text)
            start = index
        if start != len(text):
            parts.append((False, text[start:]))
        return parts

    def _find_longest_match(self, text, start):
        branch = self.user_tree
        match = None
        index = start
        while index < len(text) and text[index] in branch:
            branch = branch[text[index]]
            index = index + 1
            if "id" in branch:
                match = text[start:index]
        return match

    def _encode_bpe_segment(self, text):
        symbols = []
        for char in text:
            token_id = self.piece_to_id.get(char)
            if token_id is not None:
                symbols.append(char)
                continue
            symbols.extend(self._encode_unknown_byte_texts(char))
        symbols = merge_bpe_symbols(symbols, self.merge_ranks, self.piece_to_id)
        return [self.piece_to_id[symbol] for symbol in symbols]

    def _encode_unknown_byte_texts(self, text):
        return encode_byte_texts(text)

    def _flush_byte_buffer(self, byte_buffer):
        if not byte_buffer:
            return b""
        output = bytes(byte_buffer)
        byte_buffer.clear()
        return output


def load_tokenizer_source(path):
    path = Path(path)
    if path.suffix != ".json":
        message = "Gemma4Tokenizer now supports tokenizer.json only."
        raise ValueError(message)
    with open(str(path), encoding="utf-8") as file:
        data = json.load(file)
    model = data["model"]
    if model["type"] != "BPE":
        message = "Unsupported tokenizer model type: {}".format(model["type"])
        raise ValueError(message)
    pieces = build_piece_args(data)
    normalizer = build_normalizer_args(data.get("normalizer"))
    denormalizer = build_denormalizer_args(data.get("decoder"))
    merge_ranks = build_merge_ranks(model["merges"])
    return TokenizerSource(normalizer, denormalizer, pieces, merge_ranks)


def build_piece_args(data):
    vocab = data["model"]["vocab"]
    added = data.get("added_tokens", [])
    pieces = [None] * len(vocab)
    special_texts = {item["content"] for item in added if item["special"]}
    for text, index in vocab.items():
        piece_type = build_piece_type(text, special_texts)
        pieces[index] = PieceArgs(text, piece_type, index)
    return tuple(pieces)


def build_piece_type(text, special_texts):
    if BYTE_PATTERN.fullmatch(text):
        return PIECE_BYTE
    if text in special_texts:
        return PIECE_USER_DEFINED
    return PIECE_NORMAL


def build_piece_to_id(pieces):
    return {piece.text: piece.index for piece in pieces}


def build_normalizer_args(spec):
    if spec is None:
        return NormalizerArgs("identity", False, False, False)
    uses_metaspace = spec["type"] == "Replace"
    return NormalizerArgs("replace", False, False, uses_metaspace)


def build_denormalizer_args(spec):
    if spec is None:
        return NormalizerArgs("identity", False, False, False)
    return NormalizerArgs("replace", False, False, False)


def build_prefix_tree(pieces, piece_type):
    tree = {}
    for piece in pieces:
        if piece.type != piece_type:
            continue
        branch = tree
        for char in piece.text:
            branch = branch.setdefault(char, {})
        branch["id"] = piece.index
    return tree


def build_special_token_ids(piece_to_id, has_vision_tokens, has_audio_tokens):
    image_ids = build_media_token_ids(
        piece_to_id, has_vision_tokens, IMAGE_TOKEN_TEXTS
    )
    audio_ids = build_media_token_ids(
        piece_to_id, has_audio_tokens, AUDIO_TOKEN_TEXTS
    )
    turn_ids = tuple(piece_to_id.get(text, -1) for text in TURN_TOKEN_TEXTS)
    values = (
        piece_to_id["<bos>"],
        piece_to_id["<eos>"],
        piece_to_id["<pad>"],
        *image_ids,
        *audio_ids,
        *turn_ids,
    )
    return SpecialTokenIds(*values)


def build_media_token_ids(piece_to_id, enabled, token_texts):
    if not enabled:
        return (-1, -1, -1)
    return tuple(piece_to_id.get(text, -1) for text in token_texts)


def normalize_text(text, normalizer):
    if normalizer.remove_extra_whitespaces:
        text = SPACE_PATTERN.sub(" ", text.strip())
    if normalizer.add_dummy_prefix:
        text = " " + text
    if normalizer.escape_whitespaces or normalizer.name == "nmt_nfkc":
        text = text.replace(" ", WHITESPACE)
    return text


def denormalize_text(text, normalizer):
    if normalizer.name != "identity":
        text = text.replace(WHITESPACE, " ")
    if normalizer.add_dummy_prefix and text.startswith(" "):
        text = text[1:]
    return text


def build_merge_ranks(merges):
    return {tuple(merge): index for index, merge in enumerate(merges)}


def merge_bpe_symbols(symbols, merge_ranks, piece_to_id):
    symbols = list(symbols)
    while True:
        best = None
        for index in range(len(symbols) - 1):
            pair = (symbols[index], symbols[index + 1])
            rank = merge_ranks.get(pair)
            if rank is None:
                continue
            merged = pair[0] + pair[1]
            if merged not in piece_to_id:
                continue
            choice = (rank, index, merged)
            if best is None or choice < best:
                best = choice
        if best is None:
            return symbols
        _, index, merged = best
        symbols[index:index + 2] = [merged]


def build_byte_piece_text(byte_value):
    return "<0x{:02X}>".format(byte_value)


def build_prompt_parts(prompt):
    return ("<bos>", "<|turn>user\n", prompt, "<turn|>\n", "<|turn>model\n")


def encode_byte_texts(text):
    return [build_byte_piece_text(value) for value in text.encode()]
