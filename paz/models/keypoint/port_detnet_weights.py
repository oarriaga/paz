import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from argparse import ArgumentParser
from pathlib import Path

from keras.utils import get_file

from paz.models.keypoint.detnet import DetNet
from paz.models.keypoint.detnet import WEIGHT_PATH


DEFAULT_OUTPUT_PATH = "~/.keras/paz/models/detnet_paz_jax.weights.h5"


def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--input_shape", nargs=3, type=int, default=(128, 128, 3))
    parser.add_argument("--num_keypoints", type=int, default=21)
    return parser


def resolve_input_path(input_path):
    if input_path is not None:
        return Path(input_path).expanduser()
    filename = WEIGHT_PATH.rsplit("/", 1)[-1]
    cached_path = get_file(filename, WEIGHT_PATH, cache_subdir="paz/models")
    return Path(cached_path)


def port_weights(input_path, output_path, input_shape, num_keypoints):
    model = DetNet(
        input_shape=tuple(input_shape),
        num_keypoints=num_keypoints,
        weights=None,
    )
    model.load_weights(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(output_path)
    return output_path


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    input_path = resolve_input_path(args.input_path)
    output_path = Path(args.output_path).expanduser()
    output_path = port_weights(
        input_path,
        output_path,
        args.input_shape,
        args.num_keypoints,
    )
    print(f"Loaded weights from {input_path}")
    print(f"Saved renamed weights to {output_path}")


if __name__ == "__main__":
    main()
