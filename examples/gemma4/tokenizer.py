import json
from collections import namedtuple
from pathlib import Path
import re

MODEL_UNIGRAM = 1
MODEL_BPE = 2
MODEL_WORD = 3
MODEL_CHAR = 4

PIECE_NORMAL = 1
PIECE_UNKNOWN = 2
PIECE_CONTROL = 3
PIECE_USER_DEFINED = 4
PIECE_UNUSED = 5
PIECE_BYTE = 6

WHITESPACE = "\u2581"
SPACE_PATTERN = re.compile(r" +")
BYTE_PATTERN = re.compile(r"<0x[0-9A-Fa-f]{2}>")
MODEL_TYPES = (
    ("UNIGRAM", MODEL_UNIGRAM),
    ("BPE", MODEL_BPE),
    ("WORD", MODEL_WORD),
    ("CHAR", MODEL_CHAR),
)
PIECE_TYPES = (
    ("NORMAL", PIECE_NORMAL),
    ("UNKNOWN", PIECE_UNKNOWN),
    ("CONTROL", PIECE_CONTROL),
    ("USER_DEFINED", PIECE_USER_DEFINED),
    ("UNUSED", PIECE_UNUSED),
    ("BYTE", PIECE_BYTE),
)
IMAGE_TOKEN_TEXTS = ("<|image>", "<|image|>", "<image|>")
AUDIO_TOKEN_TEXTS = ("<|audio>", "<|audio|>", "<audio|>")
TURN_TOKEN_TEXTS = ("<|turn>", "<turn|>")
PieceArgs = namedtuple("PieceArgs", "text score type index")
NormalizerArgs = namedtuple(
    "NormalizerArgs",
    "name add_dummy_prefix remove_extra_whitespaces escape_whitespaces",
)
TrainerArgs = namedtuple(
    "TrainerArgs",
    "model_type byte_fallback unk_id bos_id eos_id pad_id",
)
TokenizerSource = namedtuple(
    "TokenizerSource",
    "normalizer denormalizer trainer pieces merge_ranks",
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
        proto,
        add_bos=False,
        add_eos=False,
        has_vision_tokens=True,
        has_audio_tokens=False,
    ):
        source = load_tokenizer_source(proto)
        self.proto = proto
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.normalizer = source.normalizer
        self.denormalizer = source.denormalizer
        self.trainer = source.trainer
        self.pieces = source.pieces
        self.id_to_piece = self.pieces
        self.piece_to_id = {piece.text: piece.index for piece in self.pieces}
        self.normal_tree = build_prefix_tree(self.pieces, PIECE_NORMAL)
        self.user_tree = build_prefix_tree(self.pieces, PIECE_USER_DEFINED)
        self.byte_to_id = build_byte_piece_map(self.pieces)
        self.merge_ranks = source.merge_ranks
        self.special_ids = build_special_token_ids(
            self.piece_to_id, has_vision_tokens, has_audio_tokens
        )
        for name in self.special_ids._fields:
            setattr(self, name, getattr(self.special_ids, name))

    def vocabulary_size(self):
        return len(self.pieces)

    def get_vocabulary(self):
        return [piece.text for piece in self.pieces]

    def id_to_token(self, token_id):
        if token_id < 0 or token_id >= self.vocabulary_size():
            size = self.vocabulary_size() - 1
            message = "`id` must be in range [0, {}].".format(size)
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
        parts = (
            "<bos>",
            "<|turn>user\n",
            prompt,
            "<turn|>\n",
            "<|turn>model\n",
        )
        return "".join(parts)

    def tokenize_generation_prompt(self, prompt):
        text = self.format_generation_prompt(prompt)
        return self.tokenize(text)

    def get_stop_token_ids(self):
        stop_ids = [self.end_token_id]
        if self.end_of_turn_token_id >= 0:
            stop_ids.append(self.end_of_turn_token_id)
        return tuple(stop_ids)

    def _tokenize_text(self, text):
        if self.trainer.model_type == MODEL_WORD:
            token_ids = self._tokenize_word_text(text)
        elif self.trainer.model_type in (MODEL_BPE, MODEL_UNIGRAM):
            token_ids = self._tokenize_piece_text(text)
        elif self.trainer.model_type == MODEL_CHAR:
            token_ids = self._tokenize_char_text(text)
        else:
            message = "Unsupported model type: {}".format(
                self.trainer.model_type
            )
            raise NotImplementedError(message)
        return self._add_boundary_tokens(token_ids)

    def _detokenize_ids(self, token_ids):
        pieces = [self.id_to_piece[token_id] for token_id in token_ids]
        if self.trainer.model_type == MODEL_WORD:
            return self._detokenize_word_pieces(pieces)
        return self._detokenize_piece_pieces(pieces)

    def _add_boundary_tokens(self, token_ids):
        output = list(token_ids)
        if self.add_bos:
            output.insert(0, self.start_token_id)
        if self.add_eos:
            output.append(self.end_token_id)
        return output

    def _tokenize_word_text(self, text):
        token_ids = []
        for piece_text in text.split():
            candidates = (piece_text, WHITESPACE + piece_text)
            self._append_piece_id(token_ids, piece_text, candidates)
        return token_ids

    def _tokenize_char_text(self, text):
        token_ids = []
        for character in text:
            self._append_piece_id(token_ids, character, (character,))
        return token_ids

    def _tokenize_piece_text(self, text):
        text = normalize_text(text, self.normalizer)
        segments = self._split_user_defined_segments(text)
        token_ids = []
        for is_user_defined, segment_text in segments:
            if is_user_defined:
                token_ids.append(self.piece_to_id[segment_text])
                continue
            if self.merge_ranks is not None:
                token_ids.extend(self._encode_bpe_segment(segment_text))
                continue
            token_ids.extend(self._encode_piece_segment(segment_text))
        return token_ids

    def _detokenize_word_pieces(self, pieces):
        text = "".join(piece.text for piece in pieces)
        text = text.replace(WHITESPACE, " ")
        return text.lstrip(" ")

    def _detokenize_piece_pieces(self, pieces):
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

    def _split_user_defined_segments(self, text):
        segments = []
        start = 0
        index = 0
        while index < len(text):
            token_text = self._find_longest_match(self.user_tree, text, index)
            if token_text is None:
                index = index + 1
                continue
            if start != index:
                segments.append((False, text[start:index]))
            segments.append((True, token_text))
            index = index + len(token_text)
            start = index
        if start != len(text):
            segments.append((False, text[start:]))
        return segments

    def _encode_bpe_segment(self, text):
        symbols = []
        for character in text:
            if character in self.piece_to_id:
                symbols.append(character)
                continue
            symbols.extend(self._encode_unknown_byte_texts(character))
        symbols = merge_bpe_symbols(symbols, self.merge_ranks, self.piece_to_id)
        return [self.piece_to_id[symbol] for symbol in symbols]

    def _encode_piece_segment(self, text):
        if not text:
            return []
        best_scores = [None] * (len(text) + 1)
        best_steps = [None] * (len(text) + 1)
        best_scores[-1] = 0.0
        for index in range(len(text) - 1, -1, -1):
            self._update_piece_step(text, index, best_scores, best_steps)
            if best_steps[index] is not None:
                continue
            next_index = index + 1
            byte_ids = self._encode_unknown_bytes(text[index])
            score = -1e9 - len(byte_ids)
            if best_scores[next_index] is not None:
                score = score + best_scores[next_index]
            best_scores[index] = score
            best_steps[index] = (next_index, tuple(byte_ids))
        token_ids = []
        index = 0
        while index < len(text):
            next_index, step_ids = best_steps[index]
            token_ids.extend(step_ids)
            index = next_index
        return token_ids

    def _update_piece_step(self, text, index, best_scores, best_steps):
        branch = self.normal_tree
        cursor = index
        while cursor < len(text) and text[cursor] in branch:
            branch = branch[text[cursor]]
            cursor = cursor + 1
            token_id = branch.get("id")
            if token_id is None or best_scores[cursor] is None:
                continue
            piece = self.id_to_piece[token_id]
            score = piece.score + best_scores[cursor]
            current_score = best_scores[index]
            if current_score is None or score > current_score:
                best_scores[index] = score
                best_steps[index] = (cursor, (token_id,))

    def _find_longest_match(self, tree, text, start):
        branch = tree
        match = None
        cursor = start
        while cursor < len(text) and text[cursor] in branch:
            branch = branch[text[cursor]]
            cursor = cursor + 1
            if "id" in branch:
                match = text[start:cursor]
        return match

    def _append_piece_id(self, token_ids, raw_text, candidates):
        for piece_text in candidates:
            token_id = self.piece_to_id.get(piece_text)
            if token_id is not None:
                token_ids.append(token_id)
                return
        token_ids.extend(self._encode_unknown_bytes(raw_text))

    def _encode_unknown_bytes(self, text):
        byte_texts = self._encode_unknown_byte_texts(text)
        return [self.piece_to_id[piece_text] for piece_text in byte_texts]

    def _encode_unknown_byte_texts(self, text):
        return [build_byte_piece_text(byte_value) for byte_value in text.encode()]

    def _flush_byte_buffer(self, byte_buffer):
        if not byte_buffer:
            return b""
        output = bytes(byte_buffer)
        byte_buffer.clear()
        return output


def load_tokenizer_source(proto):
    if is_tokenizer_json_path(proto):
        return load_hf_tokenizer(proto)
    return load_sentencepiece_tokenizer(proto)


def is_tokenizer_json_path(proto):
    if not isinstance(proto, (str, Path)):
        return False
    return Path(proto).suffix == ".json"


def load_hf_tokenizer(path):
    with open(str(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    model = data["model"]
    merge_ranks = build_merge_ranks(model["merges"])
    pieces = build_piece_args_from_json(data)
    trainer = build_trainer_args_from_json(model, pieces)
    normalizer = build_normalizer_args_from_json(data.get("normalizer"))
    denormalizer = build_denormalizer_args_from_json(data.get("decoder"))
    return TokenizerSource(
        normalizer, denormalizer, trainer, pieces, merge_ranks
    )


def load_sentencepiece_tokenizer(proto):
    model = load_sentencepiece_model(proto)
    normalizer = build_normalizer_args(model.normalizer_spec)
    denormalizer = build_normalizer_args(model.denormalizer_spec)
    trainer = build_trainer_args(model.trainer_spec)
    pieces = build_piece_args(model)
    return TokenizerSource(normalizer, denormalizer, trainer, pieces, None)


def load_sentencepiece_model(proto):
    if isinstance(proto, (str, Path)):
        proto = Path(proto).expanduser().resolve().read_bytes()
    model_class = build_sentencepiece_proto_class()
    model = model_class()
    model.ParseFromString(proto)
    return model


def build_normalizer_args(spec):
    return NormalizerArgs(
        spec.name or "identity",
        spec.add_dummy_prefix,
        spec.remove_extra_whitespaces,
        spec.escape_whitespaces,
    )


def build_trainer_args(spec):
    return TrainerArgs(
        spec.model_type,
        spec.byte_fallback,
        spec.unk_id,
        spec.bos_id,
        spec.eos_id,
        spec.pad_id,
    )


def build_piece_args(model):
    return tuple(
        PieceArgs(piece.piece, piece.score, piece.type, index)
        for index, piece in enumerate(model.pieces)
    )


def build_piece_args_from_json(data):
    vocab = data["model"]["vocab"]
    added = data.get("added_tokens", [])
    pieces = [None] * len(vocab)
    special_texts = {item["content"] for item in added if item["special"]}
    for text, index in vocab.items():
        piece_type = build_piece_type(text, special_texts)
        pieces[index] = PieceArgs(text, 0.0, piece_type, index)
    return tuple(pieces)


def build_piece_type(text, special_texts):
    if BYTE_PATTERN.fullmatch(text):
        return PIECE_BYTE
    if text in special_texts:
        return PIECE_USER_DEFINED
    return PIECE_NORMAL


def build_trainer_args_from_json(model, pieces):
    piece_to_id = {piece.text: piece.index for piece in pieces}
    unk_id = piece_to_id.get(model.get("unk_token"), -1)
    return TrainerArgs(
        MODEL_BPE,
        model.get("byte_fallback", False),
        unk_id,
        piece_to_id["<bos>"],
        piece_to_id["<eos>"],
        piece_to_id["<pad>"],
    )


def build_normalizer_args_from_json(spec):
    if spec is None:
        return NormalizerArgs("identity", False, False, False)
    uses_metaspace = spec["type"] == "Replace"
    return NormalizerArgs("replace", False, False, uses_metaspace)


def build_denormalizer_args_from_json(spec):
    if spec is None:
        return NormalizerArgs("identity", False, False, False)
    return NormalizerArgs("replace", False, False, False)


def build_prefix_tree(pieces, piece_type):
    tree = {}
    for piece in pieces:
        if piece.type != piece_type:
            continue
        branch = tree
        for character in piece.text:
            branch = branch.setdefault(character, {})
        branch["id"] = piece.index
    return tree


def build_special_token_ids(piece_to_id, has_vision_tokens, has_audio_tokens):
    image_ids = build_media_token_ids(
        piece_to_id, has_vision_tokens, IMAGE_TOKEN_TEXTS
    )
    audio_ids = build_media_token_ids(
        piece_to_id, has_audio_tokens, AUDIO_TOKEN_TEXTS
    )
    turn_ids = tuple(piece_to_id.get(token_text, -1) for token_text in TURN_TOKEN_TEXTS)
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
    return tuple(piece_to_id.get(token_text, -1) for token_text in token_texts)


def normalize_text(text, normalizer):
    if normalizer.remove_extra_whitespaces:
        text = SPACE_PATTERN.sub(" ", text.strip())
    if normalizer.add_dummy_prefix:
        text = " " + text
    use_metaspace = normalizer.escape_whitespaces
    if normalizer.name == "nmt_nfkc":
        use_metaspace = True
    if use_metaspace:
        text = text.replace(" ", WHITESPACE)
    return text


def denormalize_text(text, normalizer):
    if normalizer.name != "identity":
        text = text.replace(WHITESPACE, " ")
    if normalizer.add_dummy_prefix and text.startswith(" "):
        text = text[1:]
    return text


def build_byte_piece_map(pieces):
    return {
        int(piece.text[3:5], 16): piece.index
        for piece in pieces
        if piece.type == PIECE_BYTE and BYTE_PATTERN.fullmatch(piece.text)
    }


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
            args = (rank, index, merged)
            if best is None or args < best:
                best = args
        if best is None:
            return symbols
        _, index, merged = best
        symbols[index:index + 2] = [merged]


def build_byte_piece_text(byte_value):
    return "<0x{:02X}>".format(byte_value)


def build_sentencepiece_proto_class():
    from google.protobuf import descriptor_pb2
    from google.protobuf import descriptor_pool
    from google.protobuf import message_factory

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "sentencepiece_model.proto"
    file_proto.package = "sentencepiece"
    file_proto.syntax = "proto2"
    add_trainer_spec_descriptor(file_proto, descriptor_pb2)
    add_normalizer_descriptor(file_proto, descriptor_pb2)
    add_model_proto_descriptor(file_proto, descriptor_pb2)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    descriptor = pool.FindMessageTypeByName("sentencepiece.ModelProto")
    return message_factory.GetMessageClass(descriptor)


def add_trainer_spec_descriptor(file_proto, descriptor_pb2):
    message = file_proto.message_type.add()
    message.name = "TrainerSpec"
    add_enum(message, "ModelType", MODEL_TYPES)
    fields = (
        (
            "model_type",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            ".sentencepiece.TrainerSpec.ModelType",
        ),
        ("byte_fallback", 35, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        ("unk_id", 40, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        ("bos_id", 41, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        ("eos_id", 42, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        ("pad_id", 43, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    )
    for args in fields:
        add_field(message, *args, descriptor_pb2=descriptor_pb2)


def add_normalizer_descriptor(file_proto, descriptor_pb2):
    message = file_proto.message_type.add()
    message.name = "NormalizerSpec"
    fields = (
        ("name", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
        ("add_dummy_prefix", 3, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
        (
            "remove_extra_whitespaces",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_BOOL,
        ),
        (
            "escape_whitespaces",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_BOOL,
        ),
    )
    for args in fields:
        add_field(message, *args, descriptor_pb2=descriptor_pb2)


def add_model_proto_descriptor(file_proto, descriptor_pb2):
    message = file_proto.message_type.add()
    message.name = "ModelProto"
    piece = message.nested_type.add()
    piece.name = "SentencePiece"
    add_enum(piece, "Type", PIECE_TYPES)
    piece_fields = (
        ("piece", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
        ("score", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        (
            "type",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            ".sentencepiece.ModelProto.SentencePiece.Type",
        ),
    )
    for args in piece_fields:
        add_field(piece, *args, descriptor_pb2=descriptor_pb2)
    model_fields = (
        (
            "pieces",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".sentencepiece.ModelProto.SentencePiece",
            True,
        ),
        (
            "trainer_spec",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".sentencepiece.TrainerSpec",
        ),
        (
            "normalizer_spec",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".sentencepiece.NormalizerSpec",
        ),
        (
            "denormalizer_spec",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".sentencepiece.NormalizerSpec",
        ),
    )
    for args in model_fields:
        add_field(message, *args, descriptor_pb2=descriptor_pb2)


def add_enum(message, name, values):
    enum_type = message.enum_type.add()
    enum_type.name = name
    for value_name, number in values:
        value = enum_type.value.add()
        value.name = value_name
        value.number = number


def add_field(
    message,
    name,
    number,
    field_type,
    type_name=None,
    repeated=False,
    descriptor_pb2=None,
):
    field = message.field.add()
    field.name = name
    field.number = number
    field.type = field_type
    if repeated:
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    else:
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    if type_name is not None:
        field.type_name = type_name
