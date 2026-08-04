import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"

import paz
import scenes
import codebook

WEIGHTS = "experiments/aae.weights.h5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implicit orientation demo")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--distance", default=2.5, type=float)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--H", default=480, type=int)
    parser.add_argument("--W", default=640, type=int)
    args = parser.parse_args()

    size = (args.image_size, args.image_size)
    model = paz.models.AutoEncoder((args.image_size, args.image_size, 3))
    model.load_weights(args.weights)
    encoder = paz.models.extract_encoder(model)
    mesh = scenes.build_mesh(args.mesh)
    render = scenes.build_renderer(mesh, args.image_size, args.distance)
    poses = scenes.grid_poses(10, 10, args.distance)
    book = codebook.build_codebook(encoder, render, poses)

    def predict(image):
        crop = paz.image.normalize(paz.image.resize_opencv(image, size))
        view, pose = codebook.closest_view(encoder, crop, book)
        return paz.image.resize_opencv(view, size)

    camera = paz.Camera(args.camera)
    player = paz.VideoPlayer((args.H, args.W), predict, camera)
    player.run()
