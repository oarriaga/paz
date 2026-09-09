import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# The vocabulary projection is a 3840x262144 matmul. Next to 24 GB of
# weights there is no room for the autotuner's profiling buffers, so ask
# XLA for default algorithms instead.
flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = flags + " --xla_gpu_autotune_level=0"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paz.applications import GenerateGemma412B
from examples.gemma4.demo import chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gemma 4 12B text chat demo")
    add = parser.add_argument
    # There is no published paz artifact for 12B; convert the official
    # checkpoint with paz.models.foundation.gemma4.huggingface first.
    add("--models_path", required=True)
    add("--max_tokens", default=64, type=int)
    add("--max_prompt", default=128, type=int)
    add("--max_seq", default=256, type=int)
    args = parser.parse_args()
    generate = GenerateGemma412B(
        max_tokens=args.max_tokens, max_seq=args.max_seq,
        max_prompt=args.max_prompt, models_path=args.models_path)
    chat(generate)
