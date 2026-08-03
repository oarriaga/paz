import pytest
import torch
import numpy as np
import sys
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
sys.path.append(project_root)
rel_path = "../../../../../../examples/rf-detr_original_pytorch_implementation"
rf_detr_path = os.path.abspath(os.path.join(current_dir, rel_path))
sys.path.append(rf_detr_path)

from rfdetr import (
    RFDETRSmall,
    RFDETRMedium,
    RFDETRNano,
    RFDETRLarge,
    RFDETRBase,
    RFDETRSegNano,
    RFDETRSegSmall,
)
from rfdetr.models.matcher import build_matcher as build_torch_matcher
try:
    from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
except ImportError:
    import rfdetr.util.box_ops as box_ops
    box_cxcywh_to_xyxy = box_ops.box_cxcywh_to_xyxy
    generalized_box_iou = box_ops.generalized_box_iou

from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import (  # fmt: skip
    compute_cost_matrix,
)

from paz.models.detection.dino_v2_object_detection.models.matcher.matcher_porting_utils import (  # fmt: skip
    to_numpy, convert_to_keras, extract_matcher_config, 
    build_keras_matcher_from_config, assert_matcher_parity
)

MODELS_TO_TEST = [
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRBase,
    RFDETRLarge,
    RFDETRSegNano, 
    RFDETRSegSmall,
]

def compute_pytorch_cost_matrix(outputs, targets, matcher):
    with torch.no_grad():
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        out_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_xyxy, tgt_xyxy)

        alpha = matcher.focal_alpha
        gamma = 2.0

        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        neg_logsigmoid = -torch.nn.functional.logsigmoid(-flat_pred_logits)
        neg_focal = (1 - alpha) * (out_prob ** gamma)
        neg_cost_class = neg_focal * neg_logsigmoid
        pos_logsigmoid = -torch.nn.functional.logsigmoid(flat_pred_logits)
        pos_focal = alpha * ((1 - out_prob) ** gamma)
        pos_cost_class = pos_focal * pos_logsigmoid

        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        box_term = matcher.cost_bbox * cost_bbox
        class_term = matcher.cost_class * cost_class
        giou_term = matcher.cost_giou * cost_giou
        C = box_term + class_term + giou_term
        
        C = C.view(bs, num_queries, -1).cpu()
        return C

@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
def test_matcher_config_parity(model_class):
    _run_matcher_check(model_class, check_cost_matrix=True)

@pytest.mark.parametrize("overrides", [
    # Class only
    {"set_cost_class": 5.0, "set_cost_bbox": 0.0, "set_cost_giou": 0.0},
    {"set_cost_class": 0.0, "set_cost_bbox": 5.0, "set_cost_giou": 0.0},
    {"set_cost_class": 0.0, "set_cost_bbox": 0.0, "set_cost_giou": 5.0},
    {"group_detr": 3},
])
def test_matcher_custom_configs(overrides):
    kwargs = dict(config_overrides=overrides, check_cost_matrix=True)
    _run_matcher_check(RFDETRSmall, **kwargs)

def test_matcher_empty_targets():
    _run_matcher_check(RFDETRSmall, empty_targets=True, check_cost_matrix=False)

def build_matcher_args(model_class, config_overrides):
    try:
        rfdetr_wrapper = model_class(pretrain_weights=None)
    except Exception as error:
        message = "Skipping instantiation with pretrain_weights=None, "
        print(message + f"trying default: {error}")
        rfdetr_wrapper = model_class()
    args = rfdetr_wrapper.model.args
    for key, value in config_overrides.items():
        setattr(args, key, value)
    if getattr(args, "segmentation_head", False):
        defaults = (("mask_ce_loss_coef", 5.0), ("mask_dice_loss_coef", 5.0))
        for key, value in defaults + (("mask_point_sample_ratio", 16),):
            if not hasattr(args, key):
                setattr(args, key, value)
    return args


def build_matcher_pair(args):
    torch_matcher = build_torch_matcher(args)
    config = extract_matcher_config(args)
    keras_matcher = build_keras_matcher_from_config(config)
    class_str = f"Class={config['cost_class']}, Box={config['cost_bbox']}, "
    giou_str = f"GIoU={config['cost_giou']}, Alpha={config['focal_alpha']}"
    print(f"  Config detected: {class_str}{giou_str}")
    return torch_matcher, keras_matcher, config


def add_probe_masks(outputs_torch, targets_torch, batch_size, num_queries):
    mask_h, mask_w = 32, 32
    mask_shape = (batch_size, num_queries, mask_h, mask_w)
    outputs_torch["pred_masks"] = torch.randn(*mask_shape)
    for index in range(batch_size):
        n_boxes = targets_torch[index]["boxes"].shape[0]
        if n_boxes > 0:
            mask_size = (n_boxes, mask_h, mask_w)
            masks = torch.randint(0, 2, mask_size).float()
        else:
            masks = torch.zeros((0, mask_h, mask_w)).float()
        targets_torch[index]["masks"] = masks


def build_matcher_probe(args, empty_targets):
    batch_size = 2
    num_queries = args.num_queries * args.group_detr
    num_classes = args.num_classes
    logits = torch.randn(batch_size, num_queries, num_classes)
    boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 4))
    outputs_torch = {"pred_logits": logits, "pred_boxes": boxes}
    targets_torch = []
    for index in range(batch_size):
        n_boxes = 0 if (empty_targets and index == 0) else np.random.randint(1, 10)  # fmt: skip
        labels = torch.randint(0, num_classes, (n_boxes,)).long()
        targets_torch.append({"labels": labels, "boxes": torch.rand(n_boxes, 4)})  # fmt: skip
    if getattr(args, "segmentation_head", False):
        add_probe_masks(outputs_torch, targets_torch, batch_size, num_queries)
    return outputs_torch, targets_torch, batch_size, num_queries


def assert_cost_matrix_parity(probe, keras_pair, batch_C_torch, config):
    outputs_keras, targets_keras = keras_pair
    batch_size, num_queries = probe[2], probe[3]
    batch_C_keras = compute_cost_matrix(outputs_keras, targets_keras, **config)
    keras_np = to_numpy(batch_C_keras)
    batch_C_keras_np = keras_np.reshape(batch_size, num_queries, -1)
    print("  Verifying Cost Matrix values...")
    allclose_args = (batch_C_keras_np, to_numpy(batch_C_torch))
    kwargs = dict(rtol=0, atol=1e-4, err_msg="Cost Matrix mismatch")
    np.testing.assert_allclose(*allclose_args, **kwargs)
    print("  Cost Matrix Parity Confirmed (1e-4 tolerance).")


def assert_assignment_cost_parity(indices_torch, indices_keras, batch_C_torch):
    # Differences may arise from tie-breaking in the linear assignment
    # solver when several optimal solutions exist, so compare total cost.
    batch_C_torch_np = to_numpy(batch_C_torch)
    for index in range(len(indices_torch)):
        rows_torch = to_numpy(indices_torch[index][0])
        columns_torch = to_numpy(indices_torch[index][1])
        rows_keras = to_numpy(indices_keras[index][0])
        columns_keras = to_numpy(indices_keras[index][1])
        cost_torch = batch_C_torch_np[index][rows_torch, columns_torch].sum()
        cost_keras = batch_C_torch_np[index][rows_keras, columns_keras].sum()
        diff = abs(cost_torch - cost_keras)
        if diff > 1e-4:
            message = f"Cost mismatch at zero-tolerance! Batch {index}: "
            message += f"Torch={cost_torch}, Keras={cost_keras}, "
            raise AssertionError(message + f"Diff={diff}")
    message = "  Assignment Cost Parity Confirmed (1e-4). "
    print(message + "Mismatch purely due to tie-breaking.")


def report_index_parity(model_class, check_exact):
    name = model_class.__name__
    if not check_exact:
        message = f"  Segmentation model {name}: Skipped exact "
        print(message + "check due to random mask sampling. Structure valid.")
    else:
        print(f"  Model {name}: Exact parity confirmed for all batches.")


def assert_index_parity(model_class, indices_torch, indices_keras, check_exact, batch_C_torch):  # fmt: skip
    try:
        parity_kwargs = dict(check_exact=check_exact)
        assert_matcher_parity(indices_torch, indices_keras, **parity_kwargs)
        report_index_parity(model_class, check_exact)
    except AssertionError as error:
        if batch_C_torch is None:
            raise
        message = f"  Exact parity failed ({error}). Checking "
        print(message + "assignment cost parity (tolerance 1e-4)...")
        args = (indices_torch, indices_keras, batch_C_torch)
        assert_assignment_cost_parity(*args)


def _run_matcher_check(model_class, config_overrides=None, empty_targets=False, check_cost_matrix=False):  # fmt: skip
    config_overrides = config_overrides or {}
    print(f"\nTesting Matcher Parity for model: {model_class.__name__}")
    print(f"  Overrides: {config_overrides}")
    args = build_matcher_args(model_class, config_overrides)
    torch_matcher, keras_matcher, config = build_matcher_pair(args)
    probe = build_matcher_probe(args, empty_targets)
    outputs_torch, targets_torch = probe[0], probe[1]
    # The cost matrix is skipped for segmentation models because their mask
    # point sampling is random and would not be reproducible across backends.
    compare_costs = check_cost_matrix and not getattr(args, "segmentation_head", False)  # fmt: skip
    batch_C_torch = None
    with torch.no_grad():
        kwargs = dict(group_detr=args.group_detr)
        indices_torch = torch_matcher(outputs_torch, targets_torch, **kwargs)
        if compare_costs:
            cost_args = (outputs_torch, targets_torch, torch_matcher)
            batch_C_torch = compute_pytorch_cost_matrix(*cost_args)
    keras_pair = convert_to_keras(outputs_torch, targets_torch)
    if compare_costs:
        assert_cost_matrix_parity(probe, keras_pair, batch_C_torch, config)
    keras_kwargs = dict(group_detr=args.group_detr)
    indices_keras = keras_matcher(*keras_pair, **keras_kwargs)
    check_exact = not getattr(args, "segmentation_head", False)
    args_parity = (model_class, indices_torch, indices_keras, check_exact)
    assert_index_parity(*args_parity, batch_C_torch)


if __name__ == "__main__":
    pytest.main([__file__])
