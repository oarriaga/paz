"""Captures reference outputs of the LLM2Vec text encoder Kimodo uses.

Development tool for the parity tests; needs torch, transformers, peft
and the upstream ``kimodo`` package on ``PYTHONPATH``. It builds the
official LLM2Vec stack the way its README does: Llama-3-8B-Instruct from
an ungated mirror whose shards are verified against the sha256 of the
gated official files, the MNTP adapter merged, the supervised adapter
applied. It records token ids, pooling masks, hidden states and pooled
embeddings for a few prompts, in float32 on CPU and in bfloat16.

    PYTHONPATH=/path/to/kimodo \
        python paz/models/foundation/llm2vec/reference.py --output out.npz
"""
import argparse

import numpy as np
import torch
from huggingface_hub import model_info
from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
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


PROMPTS = ["A person walks forward.", "A person jumps up and down.",
           "Someone waves hello, then sits down."]


def capture(output_path):
    check_mirror_matches_official()
    fixtures = {"prompts": np.asarray(PROMPTS)}
    fixtures.update(capture_dtype(torch.float32, "cpu", "float32"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixtures.update(capture_dtype(torch.bfloat16, device, "bfloat16"))
    np.savez(output_path, **fixtures)


def capture_dtype(dtype, device, label):
    encoder = build_encoder(device, dtype)
    fixtures = {}
    for index, prompt in enumerate(PROMPTS):
        text = encoder.prepare_for_tokenization(f"!@#$%^&*(){prompt}")
        features = encoder.tokenize([text])
        embed_mask = features.pop("embed_mask")
        features = {key: value.to(device) for key, value in features.items()}
        with torch.no_grad():
            outputs = encoder.model(**features, output_hidden_states=True)
            features["embed_mask"] = embed_mask.to(device)
            pooled = encoder.get_pooling(features, outputs.last_hidden_state)
        prefix = f"{label}_{index}"
        fixtures[f"{prefix}_input_ids"] = features["input_ids"].cpu()
        fixtures[f"{prefix}_embed_mask"] = embed_mask
        embeddings, layer_0 = outputs.hidden_states[:2]
        fixtures[f"{prefix}_embeddings"] = embeddings.float().cpu()
        fixtures[f"{prefix}_layer_0"] = layer_0.float().cpu()
        fixtures[f"{prefix}_hidden"] = outputs.last_hidden_state.float().cpu()
        fixtures[f"{prefix}_pooled"] = pooled.float().cpu()
    return fixtures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    capture(parser.parse_args().output)
