import os
from pathlib import Path

import numpy as np
import pytest

from paz.models import Gemma4
from paz.models.foundation.gemma4.tokenizer import Gemma4Tokenizer
from paz.models.foundation.gemma4.multimodal_decoding import generate_eager

ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = ROOT / "examples" / "gemma4" / "weights"


def run_weights_test():
    # The real E2B/E4B weights are multi-GB and need CPU + large RAM, so this
    # end-to-end check is opt-in: GEMMA4_WEIGHTS_TEST=1 with a local weights dir.
    if os.environ.get("GEMMA4_WEIGHTS_TEST") != "1":
        return False
    return (WEIGHTS_DIR / "decoder_step.weights.h5").exists()


def test_gemma4_requires_models_path():
    with pytest.raises(ValueError):
        Gemma4("gemma4_2b")


@pytest.mark.skipif(not run_weights_test(), reason="set GEMMA4_WEIGHTS_TEST=1")
def test_gemma4_generates_capital_of_germany():
    models = Gemma4("gemma4_2b", models_path=str(WEIGHTS_DIR))
    tokenizer = Gemma4Tokenizer(WEIGHTS_DIR / "tokenizer.json")
    prompt = tokenizer.tokenize_generation_prompt("The capital of Germany is")
    stop = tokenizer.get_stop_token_ids()[-1]
    no_vision = np.zeros((0, models.config.hidden_dim), "float32")
    args = (models.decoder_step, models.per_layer_step, no_vision,
            models.config, prompt, [], stop, 16)
    generated = generate_eager(*args)
    assert "Berlin" in tokenizer.detokenize(generated)
