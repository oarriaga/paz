"""Serves Gemma 4 to `demo.py` on another computer.

Both computers must share a secret in the `PAZ_REMOTE_KEY` variable:

    PAZ_REMOTE_KEY=<secret> python serve.py
"""
import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import paz
from paz.applications.generators import build_gemma4_tokenizer
from paz.models import Gemma4
from paz.models.foundation.gemma4.multimodal_decoding import (
    build_text_generator)
from paz.models.foundation.gemma4.pretrained import resolve_dir
from paz.models.foundation.gemma4.tokenizer import Gemma4Tokenizer


def StreamGemma4(model_name, max_tokens, max_seq=512, max_prompt=128):
    """Builds `stream(prompt)`, which yields the answer one piece at a time.

    The answer is decoded by one jitted loop that reports each token to a
    callback, so the loop runs in a thread and its tokens cross a queue.
    """
    tokenize = Gemma4Tokenizer.tokenize_generation_prompt
    model_dir = resolve_dir(model_name, None)
    models = Gemma4(model_name, models_path=model_dir)
    tokenizer = build_gemma4_tokenizer(model_dir)
    stop_id = tokenizer.get_stop_token_ids()[-1]
    token_ids = Queue()
    finished = object()
    args = (models.model, stop_id, max_tokens, max_seq, max_prompt)
    decode = build_text_generator(*args, emit=token_ids.put)
    executor = ThreadPoolExecutor(max_workers=1)

    def stream(prompt):
        future = executor.submit(decode, tokenize(tokenizer, prompt))
        future.add_done_callback(lambda future: token_ids.put(finished))
        remaining = iter(token_ids.get, finished)
        try:
            for token_id in remaining:
                if int(token_id) != stop_id:
                    yield tokenizer.detokenize([int(token_id)])
        finally:
            # An abandoned answer must not leave tokens for the next one.
            list(remaining)
        future.result()

    return stream


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    names = ["gemma4_2b", "gemma4_4b"]
    parser.add_argument("--model_name", default="gemma4_2b", choices=names)
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    authkey = os.environ["PAZ_REMOTE_KEY"].encode()
    chat = StreamGemma4(args.model_name, args.max_tokens)
    paz.message.info("serving on port", args.port)
    paz.remote.serve({"chat": chat}, args.port, authkey)
