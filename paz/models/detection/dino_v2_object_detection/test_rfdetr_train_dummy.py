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
    from PIL import Image as PILImage
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(path, "JPEG")


def _make_dummy_coco_dataset(split_dir, num_images=4, num_classes=3):
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

# Very small settings: the point is to validate the code path, not to really
# train (the user's snippet uses epochs=15, batch_size=16).
EPOCHS = 2
BATCH_SIZE = 2
LEARNING_RATE = 1e-4


def report_environment():
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


def report_dataset(dataset_dir):
    print(f"  Dataset dir: {dataset_dir}")
    print(f"  Train annotations: {os.path.join(dataset_dir, 'train', '_annotations.coco.json')}")  # fmt: skip
    print(f"  Valid annotations: {os.path.join(dataset_dir, 'valid', '_annotations.coco.json')}")  # fmt: skip
    print()


def build_smoke_model():
    t0 = time.time()
    from paz.models.detection.dino_v2_object_detection.detr import RFDETRSmall  # fmt: skip
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
    return model


def register_epoch_callback(model):
    history = []

    def callback2(data):
        history.append(data)

    model.callbacks["on_fit_epoch_end"].append(callback2)
    print(f"  Callbacks registered: {list(model.callbacks.keys())}")
    print()
    return history


def run_training(model, dataset_dir):
    print(f"[Step 4] Starting training: epochs={EPOCHS}, "
          f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"  dataset_dir={dataset_dir}")
    t0 = time.time()
    # use_ema=False keeps things simple for a smoke test.
    keys = ("dataset_dir", "epochs", "batch_size", "lr", "use_ema", "tensorboard", "wandb", "output_dir")  # fmt: skip
    values = (dataset_dir, EPOCHS, BATCH_SIZE, LEARNING_RATE, False, False, False, os.path.join(dataset_dir, "output"))  # fmt: skip
    model.train(**dict(zip(keys, values)))
    print(f"\n  Training completed in {time.time() - t0:.1f}s")
    print()


def report_callback_results(history):
    print(f"  history length: {len(history)}")
    if len(history) >= EPOCHS:
        print(f"  PASS: Callback fired {len(history)} times "
              f"(expected >= {EPOCHS})")
    else:
        print(f"  FAIL: Callback fired only {len(history)} times "
              f"(expected >= {EPOCHS})")

    if history:
        print(f"  First epoch data keys: {sorted(history[0].keys())}")
        print(f"  Last epoch data: { {k: v for k, v in history[-1].items() if not k.startswith('best_')} }")  # fmt: skip
    print()


def report_output_artifacts(dataset_dir):
    output_dir = os.path.join(dataset_dir, "output")
    log_path = os.path.join(output_dir, "log.txt")
    ckpt_path = os.path.join(output_dir, "checkpoint.weights.h5")
    print(f"  log.txt exists:        {os.path.isfile(log_path)}")
    print(f"  checkpoint exists:     {os.path.isfile(ckpt_path)}")
    if os.path.isfile(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        print(f"  log.txt lines:         {len(lines)}")
    print()


def report_failure(error):
    print()
    print("!" * 70)
    print(f"SMOKE TEST FAILED: {type(error).__name__}: {error}")
    print("!" * 70)
    traceback.print_exc()
    sys.exit(1)


def cleanup_dataset(dataset_dir):
    print(f"\nCleaning up {dataset_dir} ...")
    shutil.rmtree(dataset_dir, ignore_errors=True)
    print("Done.")


def run_smoke_steps(dataset_dir):
    print("[Step 2] Instantiating RFDETRSmall() ...")
    model = build_smoke_model()
    print("[Step 3] Registering on_fit_epoch_end callback ...")
    history = register_epoch_callback(model)
    run_training(model, dataset_dir)
    print("[Step 5] Validating callback results ...")
    report_callback_results(history)
    print("[Step 6] Checking output artifacts ...")
    report_output_artifacts(dataset_dir)
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


def main():
    print("=" * 70)
    print("RF-DETR Small — Dummy Training Smoke Test")
    print("=" * 70)
    report_environment()
    print("[Step 1] Creating dummy COCO dataset ...")
    dataset_dir = make_dummy_dataset(num_classes=3, num_train=6, num_val=4)
    report_dataset(dataset_dir)
    try:
        run_smoke_steps(dataset_dir)
    except Exception as error:
        report_failure(error)
    finally:
        cleanup_dataset(dataset_dir)


if __name__ == "__main__":
    main()
