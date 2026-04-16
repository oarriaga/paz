import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gemma4.reference import export_from_preset
from examples.gemma4.reference import import_reference_gemma4_backbone

MODELS_DIR = Path(__file__).resolve().with_name("gemma4_models")


def export_preset(preset_name, output_dir, tokenizer_path=None):
    output_dir = Path(output_dir)
    print("Loading preset '{}'...".format(preset_name))
    config, model = export_from_preset(preset_name, output_dir)
    print("Exported config and weights to:", output_dir)
    print("  config.json:", (output_dir / "config.json").stat().st_size, "bytes")
    weights = output_dir / "model.weights.h5"
    print("  model.weights.h5:", weights.stat().st_size, "bytes")
    if tokenizer_path is not None:
        dest = output_dir / "tokenizer.spm"
        shutil.copy2(str(tokenizer_path), str(dest))
        print("  tokenizer.spm: copied from", tokenizer_path)
    print("Done. Config: {} layers, hidden_dim={}".format(
        config.num_layers, config.hidden_dim))
    return config, model


def export_from_local_backbone(backbone, output_dir, tokenizer_path=None):
    from examples.gemma4.reference import export_from_reference
    output_dir = Path(output_dir)
    config, model = export_from_reference(backbone, output_dir)
    if tokenizer_path is not None:
        dest = output_dir / "tokenizer.spm"
        shutil.copy2(str(tokenizer_path), str(dest))
    return config, model


if __name__ == "__main__":
    description = "Export Keras Hub Gemma 4 preset to paz format"
    parser = argparse.ArgumentParser(description=description)
    add = parser.add_argument
    add("preset_name", help="Keras Hub preset name (e.g. gemma4_2b)")
    add("--output_dir", default=None)
    add("--tokenizer", default=None, help="Path to tokenizer.spm")
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        output = MODELS_DIR / args.preset_name
    export_preset(args.preset_name, output, args.tokenizer)
