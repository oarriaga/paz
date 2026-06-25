import json
import os
from pathlib import Path

import numpy as np
import pytest

from paz.models import Gemma4
from paz.models.foundation.gemma4 import pretrained
from paz.models.foundation.gemma4.tokenizer import Gemma4Tokenizer
from paz.models.foundation.gemma4.multimodal_decoding import generate_eager

ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = ROOT / "examples" / "gemma4" / "weights"


def run_weights_test():
    # The real E2B/E4B weights are multi-GB and need CPU + large RAM, so this
    # end-to-end check is opt-in via GEMMA4_WEIGHTS_TEST=1 with local weights.
    if os.environ.get("GEMMA4_WEIGHTS_TEST") != "1":
        return False
    return (WEIGHTS_DIR / "decoder_step.weights.h5").exists()


def test_shard_and_reassemble_round_trips(tmp_path):
    blob = os.urandom(5_000_001)
    source = tmp_path / "decoder_step.weights.h5"
    source.write_bytes(blob)
    parts = pretrained.split_file(source, tmp_path, "gemma4_2b_decoder",
                                  part_bytes=2_000_000)
    assert parts == ["gemma4_2b_decoder.part0", "gemma4_2b_decoder.part1",
                     "gemma4_2b_decoder.part2"]
    part_paths = [tmp_path / name for name in parts]
    output = tmp_path / "reassembled.h5"
    pretrained.concatenate_parts(part_paths, output)
    assert output.read_bytes() == blob
    assert pretrained.compute_sha256(output) == pretrained.compute_sha256(source)


def test_shard_weights_writes_manifest(tmp_path):
    source_dir = tmp_path / "weights"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}")
    (source_dir / "decoder_step.weights.h5").write_bytes(os.urandom(3_000_000))
    output_dir = tmp_path / "release"
    manifest_path = pretrained.shard_weights(
        source_dir, "gemma4_2b", output_dir, part_bytes=2_000_000)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["config.json"]["parts"] == ["gemma4_2b_config.json.part0"]
    assert len(manifest["decoder_step.weights.h5"]["parts"]) == 2
    assert "sha256" in manifest["decoder_step.weights.h5"]
    decoder = manifest["decoder_step.weights.h5"]
    paths = [output_dir / asset for asset in decoder["parts"]]
    merged = output_dir / "merged.h5"
    pretrained.concatenate_parts(paths, merged)
    assert pretrained.compute_sha256(merged) == decoder["sha256"]


def test_assemble_detects_corruption(tmp_path, monkeypatch):
    (tmp_path / "part0").write_bytes(os.urandom(2048))
    monkeypatch.setattr(
        pretrained, "get_file",
        lambda asset, url, cache_subdir: str(tmp_path / "part0"))
    entry = {"parts": ["part0"], "sha256": "0" * 64}
    with pytest.raises(ValueError):
        pretrained.assemble_weights_file(tmp_path / "out.h5", entry, "sub")


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
