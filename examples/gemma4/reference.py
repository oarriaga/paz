import json
from pathlib import Path
import sys
import types

from keras import ops

from .causal_lm import Gemma4CausalLM
from .inference import Gemma4DecoderStep
from .model import TextBackboneArgs
from .model import TextIntermediates
from .model import build_text_backbone
from .model import compute_text_intermediates


def build_local_keras_hub_root():
    repos = Path(__file__).resolve().parents[3]
    keras_hub_root = repos / "keras-hub"
    if not keras_hub_root.exists():
        message = "Expected local keras-hub checkout at '{}'."
        raise FileNotFoundError(message.format(keras_hub_root))
    return keras_hub_root


def ensure_local_keras_hub():
    keras_hub_root = build_local_keras_hub_root()
    package_root = keras_hub_root / "keras_hub"
    src_root = package_root / "src"
    clear_local_keras_hub_modules()
    package = types.ModuleType("keras_hub")
    package.__path__ = [str(package_root)]
    src = types.ModuleType("keras_hub.src")
    src.__path__ = [str(src_root)]
    sys.modules["keras_hub"] = package
    sys.modules["keras_hub.src"] = src
    return keras_hub_root


def clear_local_keras_hub_modules():
    for name in list(sys.modules):
        if name == "keras_hub" or name.startswith("keras_hub."):
            sys.modules.pop(name)


def import_reference_gemma4_backbone():
    ensure_local_keras_hub()
    from keras_hub.src.models.gemma4.gemma4_backbone import (
        Gemma4Backbone,
    )
    return Gemma4Backbone


def import_reference_gemma4_tokenizer():
    ensure_local_keras_hub()
    from keras_hub.src.models.gemma4.gemma4_tokenizer import (
        Gemma4Tokenizer,
    )
    return Gemma4Tokenizer


def import_reference_sentencepiece_tokenizer():
    ensure_local_keras_hub()
    from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
        SentencePieceTokenizer,
    )
    return SentencePieceTokenizer


def build_reference_gemma4_tokenizer(
    proto_path,
    add_bos=False,
    add_eos=False,
    has_vision_tokens=True,
    has_audio_tokens=False,
):
    tokenizer = import_reference_gemma4_tokenizer()
    return tokenizer(
        proto=str(proto_path),
        add_bos=add_bos,
        add_eos=add_eos,
        has_vision_tokens=has_vision_tokens,
        has_audio_tokens=has_audio_tokens,
    )


def build_reference_sentencepiece_tokenizer(
    proto_path,
    add_bos=False,
    add_eos=False,
):
    tokenizer = import_reference_sentencepiece_tokenizer()
    args = (str(proto_path),)
    kwargs = {"add_bos": add_bos, "add_eos": add_eos}
    return tokenizer(proto=args[0], **kwargs)


def to_python(value):
    if hasattr(value, "to_list"):
        return value.to_list()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_reference_text_backbone(config):
    backbone = import_reference_gemma4_backbone()
    kwargs = config._asdict()
    # Reference backbone uses 0, not None, for hidden_size_per_layer_input
    if kwargs.get("hidden_size_per_layer_input") is None:
        kwargs["hidden_size_per_layer_input"] = 0
    # Strip paz-only fields unknown to Keras Hub backbone
    for paz_only in ("num_kv_shared_layers", "global_layer_indices"):
        kwargs.pop(paz_only, None)
    kwargs.update(
        vision_encoder=None,
        audio_encoder=None,
        name="reference_gemma4_text_backbone",
    )
    return backbone(**kwargs)


def compute_reference_text_intermediates(
        model, token_ids, padding_mask):
    hidden = model.token_embedding(token_ids)
    scale = ops.cast(model.hidden_dim ** 0.5, hidden.dtype)
    hidden = hidden * scale
    embedding_output = hidden
    block_outputs = []
    for block in model.transformer_layers:
        hidden, _ = block(hidden, padding_mask=padding_mask)
        block_outputs.append(hidden)
    final_output = model.layer_norm(hidden)
    args = (embedding_output, tuple(block_outputs), final_output)
    return TextIntermediates(*args)


def copy_text_backbone_weights(runtime_model, reference_model):
    rt_by_path = {w.path: w for w in runtime_model.weights}
    for ref_w in reference_model.weights:
        rt_path = reference_to_runtime_path(ref_w.path)
        rt_by_path[rt_path].assign(ref_w)
    return runtime_model


def reference_to_runtime_path(ref_path):
    if not ref_path.startswith("decoder_block_"):
        return ref_path
    if ref_path.endswith("/layer_scalar"):
        block = ref_path.split("/")[0]
        return "{}_layer_scalar/scale".format(block)
    parts = ref_path.split("/")
    block = parts[0]
    weight_name = parts[-1]
    layer_parts = parts[1:-1]
    layer_name = "_".join(layer_parts)
    return "{}_{}/{}".format(block, layer_name, weight_name)


def export_text_backbone_weights(config, filepath):
    reference_model = build_reference_text_backbone(config)
    runtime_model = build_text_backbone(config)
    copy_text_backbone_weights(runtime_model, reference_model)
    runtime_model.save_weights(str(filepath))
    return runtime_model, reference_model


def collect_runtime_weight_shapes(runtime_model):
    return tuple(w.shape for w in runtime_model.weights)


def collect_reference_weight_shapes(reference_model):
    return tuple(w.shape for w in reference_model.weights)


def compare_runtime_and_reference_intermediates(
    runtime_model,
    reference_model,
    token_ids,
    padding_mask,
):
    runtime = compute_text_intermediates(
        runtime_model, token_ids, padding_mask)
    reference = compute_reference_text_intermediates(
        reference_model, token_ids, padding_mask)
    return runtime, reference


def extract_text_backbone_args(reference_backbone):
    fields = TextBackboneArgs._fields
    values = {}
    for field in fields:
        values[field] = getattr(reference_backbone, field)
    return TextBackboneArgs(**values)


def export_from_reference(reference_backbone, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = extract_text_backbone_args(reference_backbone)
    save_config(config, output_dir / "config.json")
    step = Gemma4DecoderStep(config)
    copy_text_backbone_weights(step, reference_backbone)
    weights_path = output_dir / "model.weights.h5"
    step.save_weights(str(weights_path))
    return config, step


def export_from_preset(preset_name, output_dir):
    backbone_class = import_reference_gemma4_backbone()
    reference = backbone_class.from_preset(preset_name)
    return export_from_reference(reference, output_dir)


def save_config(config, filepath):
    data = config._asdict()
    with open(str(filepath), "w") as f:
        json.dump(data, f, indent=2)


def load_config(filepath):
    with open(str(filepath)) as f:
        data = json.load(f)
    if data.get("global_layer_indices") is not None:
        data["global_layer_indices"] = tuple(data["global_layer_indices"])
    return TextBackboneArgs(**data)
