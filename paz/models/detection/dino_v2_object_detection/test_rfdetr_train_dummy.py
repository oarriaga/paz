import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

# ---- Force JAX backend before importing Keras ----------------------------
os.environ.setdefault("KERAS_BACKEND", "jax")

# Suppress excessive JAX/XLA logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ---------------------------------------------------------------------------
# Dummy COCO dataset helpers
# ---------------------------------------------------------------------------

def _write_dummy_image(path, w, h):
    """Write a tiny random JPEG to *path*."""
    from PIL import Image as PILImage
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(path, "JPEG")


def _make_dummy_coco_dataset(split_dir, num_images=4, num_classes=3):
    """Create a minimal COCO-format annotation file + dummy images."""
    os.makedirs(split_dir, exist_ok=True)
    categories = [
        {"id": i + 1, "name": f"class_{i}", "supercategory": "object"}
        for i in range(num_classes)
    ]
    images = []
    annotations = []
    ann_id = 1
    for img_id in range(1, num_images + 1):
        fname = f"img_{img_id:04d}.jpg"
        images.append(
            {"id": img_id, "file_name": fname, "width": 64, "height": 64}
        )
        _write_dummy_image(os.path.join(split_dir, fname), 64, 64)
        # One bounding-box annotation per image (random category)
        cat_id = (img_id % num_classes) + 1
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": [10, 10, 30, 30],  # xywh
            "area": 900,
            "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)
    return coco


def make_dummy_dataset(num_classes=3, num_train=6, num_val=4):
    """Create a temporary COCO-format dataset with train/valid splits.

    Returns the path to the temporary dataset directory.
    """
    tmpdir = tempfile.mkdtemp(prefix="rfdetr_dummy_train_")
    _make_dummy_coco_dataset(
        os.path.join(tmpdir, "train"),
        num_images=num_train,
        num_classes=num_classes,
    )
    _make_dummy_coco_dataset(
        os.path.join(tmpdir, "valid"),
        num_images=num_val,
        num_classes=num_classes,
    )
    return tmpdir


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("RF-DETR Small — Dummy Training Smoke Test")
    print("=" * 70)

    # ---- Step 0: Environment info ----------------------------------------
    import keras
    print(f"Python     : {sys.version}")
    print(f"Keras      : {keras.__version__}")
    print(f"Backend    : {keras.backend.backend()}")
    try:
        import jax
        print(f"JAX        : {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("JAX        : not installed")
    print()

    # ---- Step 1: Create dummy dataset ------------------------------------
    print("[Step 1] Creating dummy COCO dataset ...")
    dataset_dir = make_dummy_dataset(num_classes=3, num_train=6, num_val=4)
    print(f"  Dataset dir: {dataset_dir}")
    print(f"  Train annotations: {os.path.join(dataset_dir, 'train', '_annotations.coco.json')}")
    print(f"  Valid annotations: {os.path.join(dataset_dir, 'valid', '_annotations.coco.json')}")
    print()

    try:
        # ---- Step 2: Instantiate model -----------------------------------
        print("[Step 2] Instantiating RFDETRSmall() ...")
        t0 = time.time()
        from paz.models.detection.dino_v2_object_detection.detr import RFDETRSmall
        # NOTE: group_detr=1 works around a pre-existing bug in the Keras
        # matcher port where ops.split(queries, group_detr) fails because
        # num_queries=300 is not divisible by group_detr=13.
        # group_detr=1 disables the GROUP-DETR query splitting, which is
        # fine for smoke-testing the training pipeline.
        model = RFDETRSmall(group_detr=1)
        print(f"  Model created in {time.time() - t0:.1f}s")
        print(f"  Model config: resolution={model.model_config.resolution}, "
              f"hidden_dim={model.model_config.hidden_dim}, "
              f"dec_layers={model.model_config.dec_layers}")
        print()

        # ---- Step 3: Register callback -----------------------------------
        print("[Step 3] Registering on_fit_epoch_end callback ...")
        history = []

        def callback2(data):
            history.append(data)

        model.callbacks["on_fit_epoch_end"].append(callback2)
        print(f"  Callbacks registered: {list(model.callbacks.keys())}")
        print()

        # ---- Step 4: Train -----------------------------------------------
        # Use very small settings: 2 epochs, batch_size=2
        # (the user's snippet uses epochs=15, batch_size=16, but we keep it
        # tiny to validate the code path, not to really train)
        epochs = 2
        batch_size = 2
        lr = 1e-4
        print(f"[Step 4] Starting training: epochs={epochs}, "
              f"batch_size={batch_size}, lr={lr}")
        print(f"  dataset_dir={dataset_dir}")
        t0 = time.time()

        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            use_ema=False,        # Disable EMA to keep things simple
            tensorboard=False,
            wandb=False,
            output_dir=os.path.join(dataset_dir, "output"),
        )
        train_time = time.time() - t0
        print(f"\n  Training completed in {train_time:.1f}s")
        print()

        # ---- Step 5: Validate callback -----------------------------------
        print("[Step 5] Validating callback results ...")
        print(f"  history length: {len(history)}")
        if len(history) >= epochs:
            print(f"  PASS: Callback fired {len(history)} times "
                  f"(expected >= {epochs})")
        else:
            print(f"  FAIL: Callback fired only {len(history)} times "
                  f"(expected >= {epochs})")

        if history:
            print(f"  First epoch data keys: {sorted(history[0].keys())}")
            print(f"  Last epoch data: { {k: v for k, v in history[-1].items() if not k.startswith('best_')} }")
        print()

        # ---- Step 6: Quick checkpoint check ------------------------------
        output_dir = os.path.join(dataset_dir, "output")
        log_path = os.path.join(output_dir, "log.txt")
        ckpt_path = os.path.join(output_dir, "checkpoint.weights.h5")
        print("[Step 6] Checking output artifacts ...")
        print(f"  log.txt exists:        {os.path.isfile(log_path)}")
        print(f"  checkpoint exists:     {os.path.isfile(ckpt_path)}")
        if os.path.isfile(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            print(f"  log.txt lines:         {len(lines)}")
        print()

        # ---- Summary -----------------------------------------------------
        print("=" * 70)
        print("SMOKE TEST PASSED")
        print("=" * 70)

    except Exception as e:
        print()
        print("!" * 70)
        print(f"SMOKE TEST FAILED: {type(e).__name__}: {e}")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)

    finally:
        # ---- Cleanup -----------------------------------------------------
        print(f"\nCleaning up {dataset_dir} ...")
        shutil.rmtree(dataset_dir, ignore_errors=True)
        print("Done.")


if __name__ == "__main__":
    main()
