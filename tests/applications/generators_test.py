import os
from pathlib import Path

import pytest

from paz.applications import GenerateGemma4
from paz.applications import DescribeImageGemma4

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "examples" / "gemma4" / "weights"


def run_weights_test():
    # The real weights are multi-GB and need CPU + large RAM, so the
    # end-to-end check is opt-in via GEMMA4_WEIGHTS_TEST=1 with local weights.
    if os.environ.get("GEMMA4_WEIGHTS_TEST") != "1":
        return False
    return (WEIGHTS_DIR / "decoder_step.weights.h5").exists()


def test_generate_gemma4_missing_models_path_raises():
    with pytest.raises((FileNotFoundError, OSError)):
        GenerateGemma4("gemma4_2b", models_path="/nonexistent/gemma4")


def test_describe_image_gemma4_missing_models_path_raises():
    with pytest.raises((FileNotFoundError, OSError)):
        DescribeImageGemma4("gemma4_2b", models_path="/nonexistent/gemma4")


@pytest.mark.skipif(not run_weights_test(), reason="set GEMMA4_WEIGHTS_TEST=1")
def test_generate_gemma4_on_capital_of_germany():
    generate = GenerateGemma4(
        "gemma4_2b", max_tokens=16, models_path=str(WEIGHTS_DIR))
    assert "Berlin" in generate("The capital of Germany is")
