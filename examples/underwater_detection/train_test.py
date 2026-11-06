import os
import json
import types
import functools
import importlib.util

import numpy as np
import jax.numpy as jp
import keras
import pytest

from paz.losses import detr
from paz.datasets import deepfish, ozfish

@functools.cache
def load_train():
    path = os.path.join(os.path.dirname(__file__), "train.py")
    name = "underwater_detection_train"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parameters():
    arguments = dict(variant="nano", dataset="deepfish", epochs=20)
    options = dict(patience=10, learning_rate=1e-4, ozfish_root=None)
    return dict(arguments, **options, steps_per_epoch=50, max_boxes=24)


def build_changes():
    arguments = dict(learning_rate=2e-4, dataset="deepfish_ozfish")
    return dict(arguments, steps_per_epoch=51, max_boxes=25)


def read_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def write_log(path, monitor, values, start):
    log = keras.callbacks.CSVLogger(path, append=True)
    log.on_train_begin()
    for epoch, value in enumerate(values, start):
        log.on_epoch_end(epoch, {"loss": 1.0, monitor: value})
    log.on_train_end()


def build_stage(boxes, labels, num_queries=4, num_classes=2):
    logits = np.full((1, num_queries, num_classes), -8.0, "float32")
    predicted = np.full((1, num_queries, 4), 0.5, "float32")
    for query, (box, label) in enumerate(zip(boxes, labels)):
        predicted[0, query] = box
        logits[0, query, label] = 8.0
    return np.concatenate([predicted, logits], axis=-1)


def build_targets():
    box = [0.5, 0.5, 0.2, 0.2, 1.0]
    return jp.asarray([[box, [-1.0] * 5]], "float32")


def build_groups(stages):
    groups = [np.stack(stage, axis=1) for stage in stages]
    return jp.asarray(np.stack(groups, axis=1))


def test_pin_parameters_writes_them_on_the_first_run(tmp_path):
    load_train().pin_parameters(str(tmp_path), build_parameters())
    assert read_json(tmp_path / "parameters.json") == build_parameters()


def test_pin_parameters_accepts_the_same_parameters_again(tmp_path):
    load_train().pin_parameters(str(tmp_path), build_parameters())
    load_train().pin_parameters(str(tmp_path), build_parameters())
    assert read_json(tmp_path / "parameters.json") == build_parameters()


@pytest.mark.parametrize("name", list(build_changes()))
def test_pin_parameters_rejects_a_changed_value(tmp_path, name):
    load_train().pin_parameters(str(tmp_path), build_parameters())
    changed = dict(build_parameters(), **{name: build_changes()[name]})
    with pytest.raises(ValueError):
        load_train().pin_parameters(str(tmp_path), changed)
    assert read_json(tmp_path / "parameters.json") == build_parameters()


def test_pin_parameters_rejects_a_missing_value(tmp_path):
    load_train().pin_parameters(str(tmp_path), build_parameters())
    changed = build_parameters()
    changed.pop("patience")
    with pytest.raises(ValueError):
        load_train().pin_parameters(str(tmp_path), changed)


def test_a_new_code_sha_is_logged_and_does_not_block_resume(tmp_path):
    train = load_train()
    train.pin_parameters(str(tmp_path), build_parameters())
    train.log_start(str(tmp_path), "1e0c407c")
    train.log_start(str(tmp_path), "9c354de8")
    train.pin_parameters(str(tmp_path), build_parameters())
    assert read_json(tmp_path / "parameters.json") == build_parameters()
    lines = (tmp_path / "starts.log").read_text().splitlines()
    assert len(lines) == 2
    assert "1e0c407c" in lines[0] and "9c354de8" in lines[1]


def test_validate_unfinished_refuses_a_finished_run(tmp_path):
    load_train().validate_unfinished(str(tmp_path))
    (tmp_path / "parameters.json").write_text("{}")
    load_train().validate_unfinished(str(tmp_path))
    (tmp_path / "final_ema.weights.h5").write_bytes(b"")
    with pytest.raises(ValueError):
        load_train().validate_unfinished(str(tmp_path))


def test_read_best_is_none_without_a_log(tmp_path):
    path = str(tmp_path / "log.csv")
    assert load_train().read_best(path, "val_loss", min) is None


def test_read_best_takes_the_lowest_value_across_resumes(tmp_path):
    path = str(tmp_path / "log.csv")
    write_log(path, "val_loss", [0.9, 0.7], 0)
    write_log(path, "val_loss", [0.4, 0.8], 2)
    write_log(path, "val_loss", [0.6, 0.5], 4)
    assert load_train().read_best(path, "val_loss", min) == 0.4


def test_read_best_takes_the_highest_value_with_max(tmp_path):
    path = str(tmp_path / "log.csv")
    write_log(path, "val_mAP", [0.1, 0.5, 0.3], 0)
    assert load_train().read_best(path, "val_mAP", max) == 0.5


def test_read_best_skips_missing_values(tmp_path):
    path = str(tmp_path / "log.csv")
    write_log(path, "val_loss", [float("nan"), 0.7, float("nan")], 0)
    log = keras.callbacks.CSVLogger(path, append=True)
    log.on_train_begin()
    log.on_epoch_end(3, {"loss": 1.0})
    log.on_train_end()
    assert "NA" in (tmp_path / "log.csv").read_text()
    assert load_train().read_best(path, "val_loss", min) == 0.7


def test_first_group_loss_is_the_loss_of_one_group():
    right = build_stage([[0.5, 0.5, 0.2, 0.2]], [1])
    y_pred = build_groups([[right], [right]])
    expected = float(detr.call(build_targets(), y_pred))
    actual = float(load_train().first_group_loss(build_targets(), y_pred))
    assert np.isclose(actual, expected, atol=1e-6)


def test_first_group_loss_ignores_the_other_groups():
    right = build_stage([[0.5, 0.5, 0.2, 0.2]], [1])
    wrong = build_stage([[0.1, 0.1, 0.2, 0.2]], [0])
    y_pred = build_groups([[right, wrong, wrong], [wrong, right, wrong]])
    first = float(detr.call(build_targets(), y_pred[:, :, :1]))
    actual = float(load_train().first_group_loss(build_targets(), y_pred))
    assert np.isclose(actual, first, atol=1e-6)
    assert actual < float(detr.call(build_targets(), y_pred))


def test_epoch_callback_sets_the_generator_epoch():
    generator = types.SimpleNamespace(epoch=0)
    callback = load_train().build_epoch_callback(generator)
    assert isinstance(callback, keras.callbacks.Callback)
    callback.on_epoch_begin(4)
    assert generator.epoch == 4
    callback.on_epoch_begin(9, logs={})
    assert generator.epoch == 9


def test_load_deepfish_ozfish_joins_native_deepfish_and_ozfish(monkeypatch):
    calls = []

    def load_deepfish(split):
        calls.append(("deepfish", split))
        return ["deep_0.jpg", "deep_1.jpg"], ["box_0", "box_1"]

    def load_ozfish(root, split):
        calls.append(("ozfish", root, split))
        return ["oz_0.jpg"], ["box_2"]

    def load_random_split(split):
        raise AssertionError("E15 uses the native DeepFish folders")

    monkeypatch.setattr(deepfish, "load", load_deepfish)
    monkeypatch.setattr(deepfish, "load_random_split", load_random_split)
    monkeypatch.setattr(ozfish, "load", load_ozfish)
    load = load_train().load_deepfish_ozfish
    paths, detections = load("oz_root", "validation")
    assert paths == ["deep_0.jpg", "deep_1.jpg", "oz_0.jpg"]
    assert detections == ["box_0", "box_1", "box_2"]
    assert ("deepfish", "validation") in calls and len(calls) == 2
    assert ("ozfish", "oz_root", "validation") in calls
