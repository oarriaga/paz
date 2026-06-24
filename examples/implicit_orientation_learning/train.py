import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
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
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.views) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        targets = self.views[chunk]
        inputs = [self.randomize(v, m)
                  for v, m in zip(targets, self.masks[chunk])]
        return np.stack(inputs).astype("float32"), targets

    def randomize(self, view, mask):
        image = view.copy()
        image[~mask] = self.background(view.shape)[~mask]
        image = add_occlusions(image, self.rng)
        return jitter_brightness(image, self.rng)

    def background(self, shape):
        if self.backgrounds is None:
            color = self.rng.uniform(0.0, 1.0, (1, 1, 3))
            return np.broadcast_to(color, shape).astype("float32")
        return self.backgrounds[self.rng.integers(len(self.backgrounds))]


def add_occlusions(image, rng, num_occlusions=2):
    H, W = image.shape[:2]
    for _ in range(num_occlusions):
        x, y = rng.integers(0, W), rng.integers(0, H)
        w, h = rng.integers(4, W // 3), rng.integers(4, H // 3)
        image[y:y + h, x:x + w] = rng.uniform(0.0, 1.0, 3)
    return image


def jitter_brightness(image, rng):
    return np.clip(image * rng.uniform(0.7, 1.3), 0.0, 1.0)


def load_backgrounds(path, image_size):
    if path is None:
        return None
    files = [os.path.join(path, f) for f in os.listdir(path)]
    crops = [paz.image.resize_opencv(paz.image.load(f), (image_size,) * 2)
             for f in files]
    return np.stack(crops).astype("float32") / 255.0


def render_dataset(mesh_path, num_views, image_size, distance, root, seed):
    cache = os.path.join(root, "views.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return data["views"], data["masks"]
    mesh = scenes.build_mesh(mesh_path)
    render = scenes.build_renderer(mesh, image_size, distance)
    poses = scenes.random_poses(num_views, distance, False, seed)
    views = scenes.render_views(render, poses)
    masks = (views < 0.9).any(axis=-1)
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
