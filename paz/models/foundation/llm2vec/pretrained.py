"""Download and build the LLM2Vec Llama-3-8B-Instruct text encoder.

The release bundle holds ``tokenizer.json`` and the 16 GB
``llm2vec_llama3.weights.h5`` (base model with both LoRA adapters merged),
sharded into parts below the GitHub asset cap and reassembled with the
Gemma 4 helpers. Llama 3 weights follow the Llama 3 Community License.
"""
from collections import namedtuple
import json
from pathlib import Path

import jax
from keras.utils import get_file

import paz
from paz.models.foundation.gemma4.pretrained import assemble_weights_file
from paz.models.foundation.gemma4.pretrained import shard_weights
from paz.models.foundation.llm2vec import encoding
from paz.models.foundation.llm2vec import model as encoder
from paz.models.foundation.llm2vec import tokenizer as llama_tokenizer

ALTAMIRA = "https://github.com/oarriaga/altamira-data/releases/download"
WEIGHTS_URL = ALTAMIRA + "/v0.30/"
CACHE_SUBDIR = "paz/models/llm2vec"
MODEL_NAME = "llm2vec_llama3"
WEIGHTS_NAME = "llm2vec_llama3.weights.h5"
TOKENIZER_NAME = "tokenizer.json"
LLM2Vec = namedtuple("LLM2Vec", "model variables tokenizer")


def LLM2VecLlama3(models_path=None):
    """Pretrained encoder, its jit-ready variables and tokenizer."""
    model_dir = resolve_dir(models_path)
    model = encoder.build()
    model.load_weights(str(model_dir / WEIGHTS_NAME))
    tokenizer = llama_tokenizer.load_tokenizer(model_dir / TOKENIZER_NAME)
    return LLM2Vec(model, paz.snapshot_variables(model), tokenizer)


def embed(llm2vec, text):
    return encoding.embed(llm2vec.model, llm2vec.variables, llm2vec.tokenizer,
                          text)


def resolve_dir(models_path):
    if models_path is None:
        model_dir = download_weights()
    else:
        model_dir = Path(models_path)
    return model_dir


def download_weights():
    asset = f"{MODEL_NAME}.manifest.json"
    manifest_path = get_file(asset, WEIGHTS_URL + asset,
                             cache_subdir=CACHE_SUBDIR)
    model_dir = Path(manifest_path).parent
    manifest = json.loads(Path(manifest_path).read_text())
    for filename, entry in manifest.items():
        args = (model_dir / filename, entry, CACHE_SUBDIR, WEIGHTS_URL)
        assemble_weights_file(*args)
    return model_dir


def shard_for_release(model_dir, output_dir):
    """Splits the converted bundle into release assets plus a manifest."""
    args = (model_dir, MODEL_NAME, output_dir)
    return shard_weights(*args, filenames=(TOKENIZER_NAME, WEIGHTS_NAME))


def place_on_device(llm2vec, device):
    variables = jax.tree.map(lambda leaf: jax.device_put(leaf, device),
                             llm2vec.variables)
    return llm2vec._replace(variables=variables)
