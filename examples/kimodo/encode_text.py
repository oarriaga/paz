"""Embeds prompts with Kimodo's official LLM2Vec text encoder.

This is the only torch step of the pipeline and runs in an environment
with torch, transformers, peft and the upstream ``kimodo`` package on
``PYTHONPATH``. The Llama-3-8B-Instruct base is loaded from an ungated
mirror whose safetensors shards are verified against the sha256 of the
official gated files, then the two LLM2Vec adapters are applied exactly as
the reference does. Embeddings are stored as ``{sanitized text: (1, 4096)}``
in an ``.npz`` that ``mujoco_demo.py`` reads.

    PYTHONPATH=/path/to/kimodo python examples/kimodo/encode_text.py \\
        "A person walks forward." --output prompts.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import model_info
from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
from kimodo.sanitize import sanitize_text
from peft import PeftModel
from transformers import AutoTokenizer

OFFICIAL = "meta-llama/Meta-Llama-3-8B-Instruct"
MIRROR = "NousResearch/Meta-Llama-3-8B-Instruct"
MNTP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUPERVISED = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
OFFICIAL_SHARD_HASHES = {
    "model-00001-of-00004.safetensors":
    "d8cf9c4d0dd972e1a2131bfe656235ee98221679711a3beef6d46dadf0f20b5c",
    "model-00002-of-00004.safetensors":
    "8d4782b4a69ef03845159ce1a15e272aadaaf134dc138d68f616098e8531729c",
    "model-00003-of-00004.safetensors":
    "3acdd690e65c24f42a24581b8467af98bd3ca357444580f8012aacd2bd607921",
    "model-00004-of-00004.safetensors":
    "67e9ad31c8c32abf3a55ee7fc7217b3ecb35fd3c74d98a5bd233e0e4d6964f46",
}


def check_mirror_matches_official():
    info = model_info(MIRROR, files_metadata=True)
    hashes = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    for name, expected in OFFICIAL_SHARD_HASHES.items():
        if hashes.get(name) != expected:
            raise ValueError(f"{MIRROR}/{name} differs from {OFFICIAL}")


def build_encoder(device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(MNTP)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = LlamaBiModel.from_pretrained(MIRROR, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, MNTP).merge_and_unload()
    model = PeftModel.from_pretrained(model, SUPERVISED)
    # The reference wraps prompts in the Llama-3 chat header for this name.
    model.config._name_or_path = OFFICIAL
    encoder = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512,
                      doc_max_length=400, skip_instruction=True)
    return encoder.to(device).eval()


def encode(encoder, texts, device):
    # Batch size one keeps embeddings independent of the prompt set.
    with torch.no_grad():
        embeddings = encoder.encode(texts, batch_size=1,
                                    show_progress_bar=False, device=device)
    return embeddings.float().cpu().numpy()[:, None, :]


def load_prompts(path):
    if not path.exists():
        return {}
    with np.load(path) as prompts:
        return {text: prompts[text] for text in prompts.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="+", help="natural-language prompts")
    parser.add_argument("--output", type=Path, required=True,
                        help="npz to create or extend")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float32"))
    args = parser.parse_args()

    check_mirror_matches_official()
    encoder = build_encoder(args.device, getattr(torch, args.dtype))
    texts = [sanitize_text(prompt) for prompt in args.prompts]
    embeddings = encode(encoder, texts, args.device)
    prompts = load_prompts(args.output)
    prompts.update(dict(zip(texts, embeddings)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **prompts)
    for text in texts:
        print(f"encoded {text!r}")
