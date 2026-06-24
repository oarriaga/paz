import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np

import paz
import scenes
import codebook

WEIGHTS = "experiments/aae.weights.h5"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Implicit orientation demo")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--distance", default=0.9, type=float)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--H", default=480, type=int)
    parser.add_argument("--W", default=640, type=int)
    return parser.parse_args()


def build_pipeline(args):
    model = paz.models.AutoEncoder((args.image_size, args.image_size, 3))
    model.load_weights(args.weights)
    encoder = paz.models.extract_encoder(model)
    mesh = scenes.build_mesh(args.mesh)
    render = scenes.build_renderer(mesh, args.image_size, args.distance)
    poses = scenes.grid_poses(10, 10, args.distance, False)
    book = codebook.build_codebook(encoder, render, poses)
    size = (args.image_size, args.image_size)

    def call(image):
        crop = paz.image.resize_opencv(image, size)
        crop = paz.image.normalize(np.asarray(crop, "float32"))
        view, pose = codebook.closest_view(encoder, crop, book)
        return paz.image.resize_opencv(paz.image.denormalize(view), size)

    return call


def main():
    args = parse_arguments()
    pipeline = build_pipeline(args)
    camera = paz.Camera(args.camera)
    player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
    player.run()


if __name__ == "__main__":
    main()
