import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import jax
import keras

import paz
import scenes


class OrientationSequence(keras.utils.PyDataset):
    def __init__(self, views, masks, batch_size, backgrounds=None, seed=0):
        super().__init__()
        self.views = views
        self.masks = masks
        self.batch_size = batch_size
        self.backgrounds = backgrounds
        self.key = jax.random.PRNGKey(seed)

    def __len__(self):
        return len(self.views) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        clean = self.views[chunk]
        targets = clean.astype("float32") / 255.0
        offset = index * self.batch_size
        inputs = [self.randomize(offset + arg, view, mask)
                  for arg, (view, mask) in enumerate(zip(clean, self.masks[chunk]))]  # fmt: skip
        return np.stack(inputs), targets

    def randomize(self, sample_arg, view, mask):
        key = jax.random.fold_in(self.key, sample_arg)
        randomize = paz.image.randomize_rendered_image
        image = randomize(key, view, mask, self.backgrounds)
        return np.asarray(image, "float32") / 255.0


def load_backgrounds(path, image_size):
    if path is None:
        return None
    files = [os.path.join(path, f) for f in os.listdir(path)]
    crops = [paz.image.resize_opencv(paz.image.load(f), (image_size,) * 2)
             for f in files]
    return np.stack(crops).astype("uint8")


def render_dataset(mesh_path, num_views, image_size, distance, root, seed):
    cache = os.path.join(root, "views.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return data["views"], data["masks"]
    mesh = scenes.build_mesh(mesh_path)
    render = scenes.build_renderer(mesh, image_size, distance)
    poses = scenes.random_poses(num_views, distance, False, seed)
    views = scenes.render_views(render, poses)
    masks = (views < 230).any(axis=-1)
    np.savez(cache, views=views, masks=masks)
    return views, masks


def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoEncoder AAE training")
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--backgrounds", default=None)
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--distance", default=0.9, type=float)
    parser.add_argument("--num_views", default=2000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.root, exist_ok=True)
    pack = (args.mesh, args.num_views, args.image_size, args.distance)
    views, masks = render_dataset(*pack, args.root, seed=0)
    backgrounds = load_backgrounds(args.backgrounds, args.image_size)
    sequence = OrientationSequence(views, masks, args.batch_size, backgrounds)
    model = paz.models.AutoEncoder((args.image_size, args.image_size, 3))
    optimizer = keras.optimizers.Adam(1e-3, amsgrad=True)
    model.compile(optimizer, "binary_crossentropy")
    weights = os.path.join(args.root, "aae.weights.h5")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.root, "log.csv")),
        keras.callbacks.ModelCheckpoint(weights, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau("loss", patience=10),
    ]
    model.fit(sequence, epochs=args.epochs, callbacks=callbacks)


if __name__ == "__main__":
    main()
