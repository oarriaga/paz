"""Import released GEAR-WBC ONNX weights into the ported Keras actor."""

from argparse import ArgumentParser
from pathlib import Path

import onnx
from onnx import numpy_helper

from paz.models.foundation.gear_wbc.model import build_actor


def port_weights(onnx_path):
    actor = build_actor()
    apply_actor_weights(actor, onnx.load(str(onnx_path)))
    return actor


def apply_actor_weights(actor, actor_onnx):
    initializers = load_onnx_initializers(actor_onnx)
    for name, kernel in initializers.items():
        stack, layer_index, kind = name.split(".")
        if kind == "weight":
            bias = initializers[f"{stack}.{layer_index}.bias"]
            weights = [kernel.transpose(1, 0), bias]
            actor.get_layer(f"{stack}_{layer_index}").set_weights(weights)


def load_onnx_initializers(model):
    initializers = {}
    for tensor in model.graph.initializer:
        initializers[tensor.name] = numpy_helper.to_array(tensor)
    return initializers


def save_actor(onnx_path, output_path):
    port_weights(onnx_path).save_weights(output_path)


def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--balance_onnx", required=True)
    parser.add_argument("--walk_onnx", required=True)
    parser.add_argument("--output_dir", default="~/.keras/paz/models/gear_wbc")
    return parser


def main():
    args = build_argument_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_actor(args.balance_onnx, output_dir / "gear_wbc_balance.weights.h5")
    save_actor(args.walk_onnx, output_dir / "gear_wbc_walk.weights.h5")
    print(f"Saved ported GEAR-WBC weights to {output_dir}")


if __name__ == "__main__":
    main()
