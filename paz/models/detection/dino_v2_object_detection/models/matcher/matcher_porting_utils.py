import functools
import numpy as np
import torch

IMPORT_WARNING = "Warning: Could not import KerasHungarianMatcher. Ensure project root is in python path."  # fmt: skip

try:
    from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import (  # fmt: skip
        hungarian_matcher as KerasHungarianMatcher,
    )
except ImportError:
    print(IMPORT_WARNING)
    KerasHungarianMatcher = None


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        result = t.detach().cpu().numpy()
    elif hasattr(t, "numpy"):
        result = t.numpy()
    else:
        result = np.array(t)
    return result


def convert_masks_to_keras(masks):
    if isinstance(masks, torch.Tensor):
        converted = to_numpy(masks)
    else:
        # Sparse/deferred masks arrive as a dict of component tensors.
        converted = {key: to_numpy(value) for key, value in masks.items()}
    return converted


def convert_target_to_keras(target):
    converted = {"labels": to_numpy(target["labels"])}
    converted["boxes"] = to_numpy(target["boxes"])
    if "masks" in target:
        converted["masks"] = to_numpy(target["masks"])
    return converted


def convert_to_keras(outputs_torch, targets_torch):
    outputs = {"pred_logits": to_numpy(outputs_torch["pred_logits"])}
    outputs["pred_boxes"] = to_numpy(outputs_torch["pred_boxes"])
    if "pred_masks" in outputs_torch:
        masks = outputs_torch["pred_masks"]
        outputs["pred_masks"] = convert_masks_to_keras(masks)
    targets = [convert_target_to_keras(target) for target in targets_torch]
    return outputs, targets


def extract_matcher_config(args):
    keys = ("cost_class", "cost_bbox", "cost_giou", "focal_alpha")
    values = (args.set_cost_class, args.set_cost_bbox, args.set_cost_giou, args.focal_alpha)  # fmt: skip
    config = dict(zip(keys, values))
    if getattr(args, "segmentation_head", False):
        # Inject default mask cost values when not explicitly configured
        config["cost_mask_ce"] = getattr(args, "mask_ce_loss_coef", 5.0)
        config["cost_mask_dice"] = getattr(args, "mask_dice_loss_coef", 5.0)
        ratio = getattr(args, "mask_point_sample_ratio", 16)
        config["mask_point_sample_ratio"] = ratio
    return config


def build_keras_matcher_from_config(config):
    if KerasHungarianMatcher is None:
        raise ImportError("KerasHungarianMatcher not imported.")
    keys = ("cost_class", "cost_bbox", "cost_giou", "focal_alpha", "mask_point_sample_ratio", "cost_mask_ce", "cost_mask_dice")  # fmt: skip
    values = (config["cost_class"], config["cost_bbox"], config["cost_giou"], config["focal_alpha"], config.get("mask_point_sample_ratio", 16), config.get("cost_mask_ce", 1.0), config.get("cost_mask_dice", 1.0))  # fmt: skip
    return functools.partial(KerasHungarianMatcher, **dict(zip(keys, values)))


def assert_index_pair_parity(torch_pair, keras_pair, index, check_exact):
    torch_rows, torch_columns = to_numpy(torch_pair[0]), to_numpy(torch_pair[1])
    keras_rows, keras_columns = keras_pair
    same_shape = torch_rows.shape == keras_rows.shape
    assert same_shape, f"Shape mismatch at batch index {index}"
    if check_exact:
        try:
            np.testing.assert_array_equal(torch_rows, keras_rows)
            np.testing.assert_array_equal(torch_columns, keras_columns)
        except AssertionError as error:
            message = f"Index mismatch at batch index {index}: {error}"
            raise AssertionError(message)


def assert_matcher_parity(indices_torch, indices_keras, check_exact=True):
    same_length = len(indices_torch) == len(indices_keras)
    assert same_length, "Number of batch elements matched differs"
    paired = zip(indices_torch, indices_keras)
    for index, (torch_pair, keras_pair) in enumerate(paired):
        assert_index_pair_parity(torch_pair, keras_pair, index, check_exact)
