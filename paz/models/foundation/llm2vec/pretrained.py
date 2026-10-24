"""Download and build the LLM2Vec Llama-3-8B-Instruct text encoder.

The release bundle holds ``tokenizer.json`` and the 16 GB
``llm2vec_llama3.weights.h5`` (base model with both LoRA adapters merged),
sharded into parts below the GitHub asset cap and reassembled with the
Gemma 4 helpers. Llama 3 weights follow the Llama 3 Community License.
"""
from collections import namedtuple
from pathlib import Path

import paz
from paz.models.foundation.gemma4.pretrained import download_bundle
from paz.models.foundation.gemma4.pretrained import shard_weights
from paz.models.foundation.llm2vec import MODEL_NAME, TOKENIZER_NAME
from paz.models.foundation.llm2vec import WEIGHTS_NAME, encoding
from paz.models.foundation.llm2vec import model as encoder
from paz.models.foundation.llm2vec import tokenizer as llama_tokenizer

ALTAMIRA = "https://github.com/oarriaga/altamira-data/releases/download"
WEIGHTS_URL = ALTAMIRA + "/v0.30/"
CACHE_SUBDIR = "paz/models/llm2vec"
LLM2Vec = namedtuple("LLM2Vec", "forward variables tokenizer")


def LLM2VecLlama3(models_path=None):
    """Pretrained encoder as a jitted forward, variables and tokenizer."""
    model_dir = resolve_dir(models_path)
    model = encoder.build()
    model.load_weights(str(model_dir / WEIGHTS_NAME))
    tokenizer = llama_tokenizer.load_tokenizer(model_dir / TOKENIZER_NAME)
    args = (encoding.build_forward(model), paz.snapshot_variables(model),
            tokenizer)
    return LLM2Vec(*args)


def embed(llm2vec, text):
    args = (llm2vec.forward, llm2vec.variables, llm2vec.tokenizer, text)
    return encoding.embed(*args)


def resolve_dir(models_path):
    if models_path is None:
        model_dir = download_bundle(MODEL_NAME, WEIGHTS_URL, CACHE_SUBDIR)
    else:
        model_dir = Path(models_path)
    return model_dir


def shard_for_release(model_dir, output_dir):
    """Splits the converted bundle into release assets plus a manifest."""
    args = (model_dir, MODEL_NAME, output_dir)
    return shard_weights(*args, filenames=(TOKENIZER_NAME, WEIGHTS_NAME))
