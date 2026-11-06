"""Stage-one RF-DETR training on DeepFish, or DeepFish with OzFish.

Experiments 10 and 11 are ``--dataset deepfish`` with ``--variant nano`` or
``large``: the pooled 80/20 DeepFish split, the best checkpoint by the
first query group's validation loss and early stopping. Experiment 15 is
``--dataset deepfish_ozfish --variant large``: the native DeepFish folders
plus OzFish, no early stopping, and ``best_total.weights.h5``, the better
of the best raw and the best EMA weights by validation COCO mAP.

A run lives in one fixed ``--run_dir``. Resubmitting the same command
resumes it from the last finished epoch.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from datetime import datetime

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import keras

import paz
from paz.datasets import deepfish, ozfish
from paz.models import TrainableRFDETRNano, TrainableRFDETRLarge

NUM_CLASSES = 1
LOSS_MONITOR = "val_first_group_loss"
VARIANTS = dict(nano=TrainableRFDETRNano, large=TrainableRFDETRLarge)


def validate_unfinished(run_dir):
    if os.path.exists(os.path.join(run_dir, "final_ema.weights.h5")):
        raise ValueError(f"{run_dir} is finished; start a new --run_dir")


def load_deepfish_ozfish(ozfish_root, split):
    image_paths, detections = deepfish.load(split)
    ozfish_paths, ozfish_detections = ozfish.load(ozfish_root, split)
    return image_paths + ozfish_paths, detections + ozfish_detections


def build_model(variant, num_groups):
    source = VARIANTS[variant](NUM_CLASSES, 1)
    weights = paz.models.detection.rf_detr.download_weights(source)
    source.load_weights(weights, skip_mismatch=True)
    model = VARIANTS[variant](NUM_CLASSES, num_groups)
    # the weights file is keyed by position, so load one group, copy by name
    copy_weights(source, model)
    paz.models.detection.rf_detr.reset_class_heads(model)
    return model


def copy_weights(source, model):
    for layer in source.layers:
        if layer.weights:
            model.get_layer(layer.name).set_weights(layer.get_weights())


def validate_steps(steps_per_epoch, num_batches):
    if steps_per_epoch > num_batches:
        raise ValueError(f"--steps_per_epoch is above {num_batches} batches")


def pin_parameters(run_dir, parameters):
    path = os.path.join(run_dir, "parameters.json")
    parameters = json.loads(json.dumps(parameters))
    if os.path.exists(path):
        validate_parameters(read_json(path), parameters)
    else:
        paz.file.write_json(parameters, path)


def validate_parameters(pinned, parameters):
    changed = set(pinned.items()) ^ set(parameters.items())
    if changed:
        names = sorted({name for name, _ in changed})
        raise ValueError(f"Parameters differ from parameters.json: {names}")


def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def log_start(run_dir, code_sha):
    start = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(run_dir, "starts.log"), "a") as log_file:
        log_file.write(f"{start} {code_sha}\n")


def read_code_sha():
    try:
        suffix = "-dirty" if run_git("status", "--porcelain") else ""
        code_sha = run_git("rev-parse", "HEAD") + suffix
    except (OSError, subprocess.CalledProcessError):
        code_sha = "unknown"
    return code_sha


def run_git(*arguments):
    folder = os.path.dirname(os.path.abspath(__file__))
    kwargs = dict(capture_output=True, text=True, check=True, cwd=folder)
    return subprocess.run(["git", *arguments], **kwargs).stdout.strip()


def build_optimizer(schedule, weight_decay, clipnorm, accumulation, momentum):
    steps = accumulation if accumulation > 1 else None
    kwargs = dict(weight_decay=weight_decay, global_clipnorm=clipnorm)
    kwargs = dict(kwargs, gradient_accumulation_steps=steps, use_ema=True)
    kwargs = dict(kwargs, ema_momentum=momentum)
    # every rate scale is one: the reference scales gradients before Adam,
    # which divides each gradient by its own size, so its decay cancels
    optimizer = paz.optimizers.LayerwiseAdamW({}, schedule, **kwargs)
    optimizer.exclude_from_weight_decay(var_names=("bias", "gamma", "beta"))
    return optimizer


def first_group_loss(y_true, y_pred):
    return paz.losses.detr.call(y_true, y_pred[:, :, :1])


def build_epoch_callback(generator):
    def set_epoch(epoch, logs=None):
        generator.epoch = epoch
    return keras.callbacks.LambdaCallback(on_epoch_begin=set_epoch)


def build_loss_checkpoint(run_dir, csv_path):
    path = os.path.join(run_dir, "best.weights.h5")
    best = read_best(csv_path, LOSS_MONITOR, min)
    kwargs = dict(save_best_only=True, save_weights_only=True, mode="min")
    kwargs = dict(kwargs, initial_value_threshold=best)
    return keras.callbacks.ModelCheckpoint(path, LOSS_MONITOR, **kwargs)


def build_early_stopping(patience):
    # restoring raw best weights would overwrite the finalized EMA
    kwargs = dict(patience=patience, min_delta=1e-3, mode="min")
    kwargs = dict(kwargs, restore_best_weights=False)
    return keras.callbacks.EarlyStopping(LOSS_MONITOR, **kwargs)


class ReplayEarlyStopping(keras.callbacks.Callback):
    """Feeds the logged losses back into early stopping after a resume."""

    def __init__(self, stop, csv_path):
        super().__init__()
        self.stop, self.csv_path = stop, csv_path

    def on_train_begin(self, logs=None):
        self.losses = dict(read_logs(self.csv_path, LOSS_MONITOR))

    def on_epoch_begin(self, epoch, logs=None):
        # the backup sets the start epoch only after every on_train_begin
        earlier = [key for key in sorted(self.losses) if key < epoch]
        for logged_epoch in earlier:
            logged = {LOSS_MONITOR: self.losses[logged_epoch]}
            self.stop.on_epoch_end(logged_epoch, logged)
        self.losses = {}


def build_best_total(model, validation_data, max_boxes, run_dir, csv_path):
    detector = paz.models.detection.rf_detr.build_detector(model)
    detect = paz.applications.detectors.RFDETR(detector, 0.0, None)
    evaluate = paz.partial(compute_mAP, detect, *validation_data, max_boxes)
    best_regular = read_best(csv_path, "val_regular_mAP", max) or 0.0
    best_ema = read_best(csv_path, "val_ema_mAP", max) or 0.0
    return BestTotal(evaluate, run_dir, best_regular, best_ema)


def compute_mAP(detect, image_paths, detections, max_boxes):
    args = (detect, image_paths, detections, NUM_CLASSES)
    return paz.evaluation.compute_COCO_mAP(*args, max_objects=max_boxes)["mAP"]


class BestTotal(keras.callbacks.Callback):
    """Keeps the best raw and the best EMA weights by validation mAP.

    Training ends by copying the better of the two to best_total, as the
    reference keeps its checkpoint_best_total.
    """

    def __init__(self, evaluate, run_dir, best_regular, best_ema):
        super().__init__()
        self.evaluate, self.run_dir = evaluate, run_dir
        self.best_regular, self.best_ema = best_regular, best_ema
        self.swap = keras.callbacks.SwapEMAWeights()

    def set_model(self, model):
        super().set_model(model)
        self.swap.set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        self.best_regular = self.score(logs, "regular", self.best_regular)
        self.swap.on_test_begin()
        self.best_ema = self.score(logs, "ema", self.best_ema)
        self.swap.on_test_end()

    def score(self, logs, name, best):
        mAP = self.evaluate()
        logs[f"val_{name}_mAP"] = mAP
        if mAP > best:
            self.model.save_weights(self.build_path(f"best_{name}"))
        return max(mAP, best)

    def on_train_end(self, logs=None):
        name = "ema" if self.best_ema > self.best_regular else "regular"
        source = self.build_path(f"best_{name}")
        if os.path.exists(source):
            shutil.copy2(source, self.build_path("best_total"))
        else:
            print("No best_total: validation mAP never rose above 0")

    def build_path(self, name):
        return os.path.join(self.run_dir, f"{name}.weights.h5")


def save_final_ema(model, run_dir, backup_path):
    temporary_path = os.path.join(run_dir, "final_ema.tmp.weights.h5")
    model.save_weights(temporary_path)
    # a rename is atomic, so a kill never leaves a truncated final file
    os.replace(temporary_path, os.path.join(run_dir, "final_ema.weights.h5"))
    shutil.rmtree(backup_path)


def read_best(csv_path, monitor, select):
    values = [value for _, value in read_logs(csv_path, monitor)]
    return select(values) if values else None


def read_logs(csv_path, monitor):
    if os.path.exists(csv_path):
        rows = read_rows(csv_path)
    else:
        rows = []
    logs = [(int(row["epoch"]), parse_value(row, monitor)) for row in rows]
    return [(epoch, value) for epoch, value in logs if not math.isnan(value)]


def read_rows(csv_path):
    with open(csv_path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return rows


def parse_value(row, monitor):
    # CSVLogger writes NA for a metric missing from an epoch
    text = row.get(monitor, "NA")
    return math.nan if text == "NA" else float(text)


if __name__ == "__main__":
    # imported here so pytest never resolves another example's generator.py
    from generator import TrainGenerator, ValidationGenerator
    from generator import validate_labels

    datasets = ["deepfish", "deepfish_ozfish"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nano", choices=list(VARIANTS))
    parser.add_argument("--dataset", default="deepfish", choices=datasets)
    parser.add_argument("--ozfish_root", default=None)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--clipnorm", default=0.1, type=float)
    parser.add_argument("--ema_momentum", default=0.993, type=float)
    parser.add_argument("--num_groups", default=13, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=None, type=int)
    parser.add_argument("--validation_steps", default=None, type=int)
    args = parser.parse_args()
    if args.dataset == "deepfish_ozfish" and args.ozfish_root is None:
        parser.error("--dataset deepfish_ozfish needs --ozfish_root")

    paz.directory.make(args.run_dir)
    validate_unfinished(args.run_dir)
    merged = paz.partial(load_deepfish_ozfish, args.ozfish_root)
    loaders = dict(deepfish=deepfish.load_random_split, deepfish_ozfish=merged)
    train_data = loaders[args.dataset]("train")
    validation_data = loaders[args.dataset]("validation")
    validate_labels(train_data[1], NUM_CLASSES)
    validate_labels(validation_data[1], NUM_CLASSES)

    keras.utils.set_random_seed(args.seed)
    model = build_model(args.variant, args.num_groups)
    size = model.input_shape[1]
    key = jax.random.PRNGKey(args.seed)
    copy_indices = [0] * len(train_data[0])
    # one worker: prefetching would serve the previous epoch's batches
    train_args = (key, *train_data, copy_indices, args.batch_size, size, 1)
    train_generator = TrainGenerator(*train_args)
    validation_args = (*validation_data, args.batch_size, size, 1)
    validation_generator = ValidationGenerator(*validation_args)

    steps_per_epoch = args.steps_per_epoch or len(train_generator)
    validate_steps(steps_per_epoch, len(train_generator))
    max_boxes = train_generator.max_boxes
    parameters = dict(vars(args), steps_per_epoch=steps_per_epoch)
    # the file lives in the run directory, so pinning its spelling adds
    # nothing and would refuse a relative versus absolute resubmit
    del parameters["run_dir"]
    # None for E10/E11; any spelling of the same folder resumes
    ozfish_root = args.ozfish_root and os.path.abspath(args.ozfish_root)
    parameters = dict(parameters, max_boxes=max_boxes, ozfish_root=ozfish_root)
    pin_parameters(args.run_dir, parameters)
    log_start(args.run_dir, read_code_sha())

    decay_steps = max(steps_per_epoch * args.epochs, 1)
    schedule_args = (args.learning_rate, decay_steps)
    schedule = keras.optimizers.schedules.CosineDecay(*schedule_args)
    decay_args = (schedule, args.weight_decay, args.clipnorm)
    step_args = (args.accumulation_steps, args.ema_momentum)
    optimizer = build_optimizer(*decay_args, *step_args)
    # the Hungarian matching is a pure_callback, traced into the jitted step
    losses, metrics = paz.losses.detr.call, [first_group_loss]
    model.compile(optimizer, losses, metrics=metrics, jit_compile=True)

    csv_path = os.path.join(args.run_dir, "log.csv")
    backup_path = os.path.join(args.run_dir, "backup")
    plot_path = os.path.join(args.run_dir, "metrics.png")
    log = keras.callbacks.CSVLogger(csv_path, append=True)
    selects_by_loss = args.dataset == "deepfish"
    if selects_by_loss:
        best = build_loss_checkpoint(args.run_dir, csv_path)
        stop = build_early_stopping(args.patience)
        replay = ReplayEarlyStopping(stop, csv_path)
        best_callbacks = [best, log, stop, replay]
    else:
        max_objects = validation_generator.max_boxes
        best_args = (validation_data, max_objects, args.run_dir, csv_path)
        best_callbacks = [build_best_total(model, *best_args), log]
    callbacks = [build_epoch_callback(train_generator)] + best_callbacks
    callbacks.append(paz.callbacks.PlotMetrics(plot_path))
    # last: an epoch is backed up only once its log row and best are saved;
    # kept until final_ema is written, so a late kill never restarts the run,
    # and doubled, so a kill while writing it falls back to the previous one
    kwargs = dict(delete_checkpoint=False, double_checkpoint=True)
    callbacks.append(keras.callbacks.BackupAndRestore(backup_path, **kwargs))

    # the generator shuffles by seed and epoch; Keras would reshuffle it
    kwargs = dict(epochs=args.epochs, callbacks=callbacks, shuffle=False)
    kwargs = dict(kwargs, steps_per_epoch=args.steps_per_epoch)
    kwargs = dict(kwargs, validation_data=validation_generator)
    kwargs = dict(kwargs, validation_steps=args.validation_steps)
    model.fit(train_generator, **kwargs)
    # fit has already finalized the EMA into the weights; never do it twice
    save_final_ema(model, args.run_dir, backup_path)
