"""Data generator for RF-DETR / LWDETR object detection training.

Provides a Keras PyDataset-based generator with threaded prefetching
for efficient data loading during training.  Overlaps I/O (image
loading, decoding, augmentation) with GPU computation so the
accelerator stays fed.
"""
import math
import queue
import threading

import numpy as np
from keras.utils import PyDataset


# ======================================================================
# Augmentation (duplicated from training_helpers to keep generator
# self-contained and importable without heavy dependencies)
# ======================================================================


def _augment_pipeline2(image, target, rng):
    """Apply pipeline2-style augmentations for DETR training.

    - Random horizontal flip (p=0.5) with box adjustment
    - Random brightness jitter (p=0.8)
    - Random contrast jitter (p=0.8)
    - Random saturation jitter (p=0.5)

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), float32 in [0, 1]
    target : dict with 'boxes' (N, 4) in cxcywh normalised and 'labels'
    rng : np.random.RandomState

    Returns
    -------
    image, target : augmented copies
    """
    boxes = target["boxes"].copy()
    labels = target["labels"].copy()

    # Horizontal flip
    if rng.rand() < 0.5:
        image = image[:, ::-1, :].copy()
        if len(boxes) > 0:
            boxes[:, 0] = 1.0 - boxes[:, 0]

    # Brightness
    if rng.rand() < 0.8:
        factor = rng.uniform(0.7, 1.3)
        image = np.clip(image * factor, 0.0, 1.0)

    # Contrast
    if rng.rand() < 0.8:
        gray_mean = image.mean()
        factor = rng.uniform(0.7, 1.3)
        image = np.clip(gray_mean + factor * (image - gray_mean), 0.0, 1.0)

    # Saturation
    if rng.rand() < 0.5:
        gray = np.mean(image, axis=-1, keepdims=True)
        factor = rng.uniform(0.7, 1.3)
        image = np.clip(gray + factor * (image - gray), 0.0, 1.0)

    image = image.astype(np.float32)
    return image, {"boxes": boxes, "labels": labels}


# ======================================================================
# Data Generator
# ======================================================================


class DetectionDataGenerator(PyDataset):
    """Keras PyDataset for object detection with prefetch support.

    Loads images and variable-length detection targets in batches.
    Supports per-epoch shuffling with deterministic seeding and
    thread-safe augmentation (each batch gets its own RNG).

    Usage with custom training loop::

        gen = DetectionDataGenerator(dataset, train_indices, batch_size=16,
                                     augmentation='pipeline2', seed=42)
        for epoch in range(num_epochs):
            gen.set_epoch(epoch)
            for images, targets in prefetch_iterator(gen):
                train_step(images, targets)

    Usage with ``model.fit()``::

        gen = DetectionDataGenerator(dataset, train_indices, batch_size=16,
                                     workers=4, max_queue_size=10)
        model.fit(gen, epochs=50)

    Parameters
    ----------
    dataset : DeepFishDataset
        Dataset supporting ``__getitem__(idx) -> (image_np, target_dict)``.
    indices : list[int]
        Indices into the dataset to iterate over.
    batch_size : int
    augmentation : str or None
        ``'pipeline2'`` for horizontal flip + color jitter, ``None``
        for no augmentation.
    seed : int
        Base random seed for shuffling and augmentation.
    shuffle : bool
        Reshuffle indices each epoch for training.  Disable for
        validation / evaluation.
    workers : int
        Number of background workers for Keras ``model.fit()`` loading.
        Has no effect when used with ``prefetch_iterator()``.
    max_queue_size : int
        Queue depth for Keras ``model.fit()`` prefetching.
    """

    def __init__(
        self,
        dataset,
        indices,
        batch_size,
        augmentation=None,
        seed=42,
        shuffle=True,
        workers=0,
        max_queue_size=10,
    ):
        super().__init__(
            workers=workers,
            use_multiprocessing=False,
            max_queue_size=max_queue_size,
        )
        self.dataset = dataset
        self._original_indices = list(indices)
        self.indices = list(indices)
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self._seed = seed
        self._epoch = 0

        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(self.indices)

    # ----- PyDataset interface ------------------------------------------

    def __len__(self):
        """Number of batches per epoch."""
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, batch_idx):
        """Load one batch by index.

        Returns
        -------
        images_np : np.ndarray, shape (B, H, W, 3), float32
        targets : list[dict]
            Each dict has ``'boxes'`` (N, 4) and ``'labels'`` (N,).
        """
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]

        # Deterministic per-batch RNG (thread-safe — no shared state)
        rng = np.random.RandomState(
            self._seed + self._epoch * 100000 + batch_idx
        )

        images, targets = [], []
        for idx in batch_indices:
            img, tgt = self.dataset[idx]
            if self.augmentation == "pipeline2":
                img, tgt = _augment_pipeline2(img, tgt, rng)
            images.append(img)
            targets.append(tgt)

        images_np = np.stack(images, axis=0).astype("float32")
        return images_np, targets

    def on_epoch_end(self):
        """Called by Keras at the end of each epoch (``model.fit``)."""
        self._epoch += 1
        if self.shuffle:
            self.indices = list(self._original_indices)
            rng = np.random.RandomState(self._seed + self._epoch)
            rng.shuffle(self.indices)

    # ----- Custom training loop helpers ---------------------------------

    def set_epoch(self, epoch):
        """Set epoch for deterministic shuffling + augmentation.

        Call this at the start of each epoch in a custom training loop
        (not needed with ``model.fit`` — ``on_epoch_end`` handles it).
        """
        self._epoch = epoch
        if self.shuffle:
            self.indices = list(self._original_indices)
            rng = np.random.RandomState(self._seed + self._epoch)
            rng.shuffle(self.indices)


# ======================================================================
# Threaded prefetching iterator
# ======================================================================


class _ErrorWrapper:
    """Sentinel wrapper so exceptions travel through the queue safely."""
    __slots__ = ("exc",)

    def __init__(self, exc):
        self.exc = exc


def prefetch_iterator(generator, max_prefetch=4):
    """Create a prefetching iterator over a DetectionDataGenerator.

    A background thread loads batches into a bounded queue while the
    main thread consumes them.  This overlaps data preparation (I/O,
    decoding, augmentation) with model computation (GPU forward /
    backward), reducing idle time on the accelerator.

    Parameters
    ----------
    generator : DetectionDataGenerator
        Must support ``__len__`` and ``__getitem__``.
    max_prefetch : int
        Maximum number of batches to buffer ahead.

    Yields
    ------
    images_np : np.ndarray, shape (B, H, W, 3), float32
    targets : list[dict]
    """
    buf = queue.Queue(maxsize=max_prefetch)
    _sentinel = object()

    def _producer():
        try:
            for i in range(len(generator)):
                buf.put(generator[i])
        except Exception as exc:
            buf.put(_ErrorWrapper(exc))
        finally:
            buf.put(_sentinel)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    try:
        while True:
            item = buf.get()
            if item is _sentinel:
                break
            if isinstance(item, _ErrorWrapper):
                raise item.exc
            yield item
    finally:
        thread.join(timeout=10.0)
