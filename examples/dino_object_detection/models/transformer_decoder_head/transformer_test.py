import pytest
import numpy as np
import torch
import torch.nn as nn
import os
import sys

# -------------------------------------------------------------------------
# 0. Environment Setup (Run only once)
# -------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["bat"] = "1"
torch.set_num_threads(1)  # Disable multi-threading noise
torch.manual_seed(42)  # Seed PyTorch
torch.use_deterministic_algorithms(True)  # Enforce deterministic ops
# Add project root to path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import keras
from keras import layers

np.random.seed(42)

# --- Project Imports ---
# Ensure these filenames match exactly what is on your disk
from examples.dino_object_detection.models.transformer_decoder_head.torch_transformer_for_testing import (
    Transformer as Transformer_PyTorch,
    MLP as MLP_PyTorch,
    gen_sineembed_for_position as gen_sineembed_for_position_pt,
)
from examples.dino_object_detection.models.transformer_decoder_head.transformer_kerass import (
    Transformer as Transformer_Keras,
)
from examples.dino_object_detection.models.transformer_decoder_head.transformer_kerass import (
    MLP as MLP_Keras,
    TransformerDecoderLayer as TransformerDecoderLayer_Keras,
    TransformerDecoder as TransformerDecoder_Keras,
    MSDeformAttn as MSDeformAttn_Keras,
)

from examples.dino_object_detection.models.transformer_decoder_head.torch_transformer_for_testing import (
    MSDeformAttn as MSDeformAttn_PyTorch,
    TransformerDecoderLayer as TransformerDecoderLayer_PyTorch,
    TransformerDecoder as TransformerDecoder_PyTorch,
)

# =============================================================================
# 1. HELPER FUNCTIONS (INLINED TO FIX IMPORT ERROR)
# =============================================================================


def check_parity(layer_idx, name, pt_tensor, k_tensor, tol=1e-4):
    """Checks parity between PyTorch and Keras tensors."""
    if isinstance(pt_tensor, torch.Tensor):
        pt_val = pt_tensor.detach().cpu().numpy()
    else:
        pt_val = pt_tensor

    k_val = np.array(k_tensor)

    # Handle shape mismatch gracefully for debugging
    if pt_val.shape != k_val.shape:
        print(f"   [Shape Mismatch] PT: {pt_val.shape} | Keras: {k_val.shape}")
        if pt_val.size == k_val.size:
            pt_val = pt_val.reshape(k_val.shape)

    diff = np.abs(pt_val - k_val)
    mae = np.mean(diff)
    max_diff = np.max(diff)

    status = "PASS" if mae < tol else "FAIL"
    print(
        f"{layer_idx:<5} | {name:<20} | {mae:.8f}     | {max_diff:.8f}     | {status}"
    )
    return status


def to_keras_mha_input_kernel(w, n_head):
    w = w.T
    d_model = w.shape[0]
    head_dim = w.shape[1] // n_head
    return w.reshape(d_model, n_head, head_dim)


def to_keras_mha_bias(b, n_head):
    head_dim = b.shape[0] // n_head
    return b.reshape(n_head, head_dim)


def to_keras_mha_output_kernel(w, n_head):
    w = w.T
    d_model_out = w.shape[1]
    head_dim = w.shape[0] // n_head
    return w.reshape(n_head, head_dim, d_model_out)


def transfer_linear_weights(pt_layer, k_layer):
    with torch.no_grad():
        k_layer.set_weights(
            [
                pt_layer.weight.detach().numpy().T,
                pt_layer.bias.detach().numpy(),
            ]
        )


def transfer_norm_weights(pt_layer, k_layer):
    with torch.no_grad():
        k_layer.set_weights(
            [
                pt_layer.weight.detach().numpy(),
                pt_layer.bias.detach().numpy(),
            ]
        )


def transfer_mlp_weights(pt_mlp, k_mlp):
    with torch.no_grad():
        for i in range(pt_mlp.num_layers):
            pt_w = pt_mlp.layers[i].weight.detach().numpy()
            pt_b = pt_mlp.layers[i].bias.detach().numpy()
            k_mlp.mlp_layers.layers[i].set_weights([pt_w.T, pt_b])


def transfer_decoder_layer_weights(pt_layer, k_layer, *args):
    """
    Handles variable arguments to support conflicting calls in the test file.
    Case 1: (pt_layer, k_layer, sa_nhead) -> 3 args
    Case 2: (pt_layer, k_layer, d_model, sa_nhead, ca_nhead) -> 5 args
    """
    if len(args) == 1:
        # Case 1: args = (sa_nhead,)
        sa_nhead = args[0]
    elif len(args) == 3:
        # Case 2: args = (d_model, sa_nhead, ca_nhead)
        sa_nhead = args[1]
    else:
        raise ValueError(f"Unexpected number of arguments: {len(args)}")

    with torch.no_grad():
        # Self Attention
        pt_sa = pt_layer.self_attn
        qkv_w = pt_sa.in_proj_weight.detach().numpy()
        qkv_b = pt_sa.in_proj_bias.detach().numpy()
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)
        out_w = pt_sa.out_proj.weight.detach().numpy()
        out_b = pt_sa.out_proj.bias.detach().numpy()

        k_layer.self_attn.set_weights(
            [
                to_keras_mha_input_kernel(q_w, sa_nhead),
                to_keras_mha_bias(q_b, sa_nhead),
                to_keras_mha_input_kernel(k_w, sa_nhead),
                to_keras_mha_bias(k_b, sa_nhead),
                to_keras_mha_input_kernel(v_w, sa_nhead),
                to_keras_mha_bias(v_b, sa_nhead),
                to_keras_mha_output_kernel(out_w, sa_nhead),
                out_b,
            ]
        )

        # Norms
        transfer_norm_weights(pt_layer.norm1, k_layer.norm1)
        transfer_norm_weights(pt_layer.norm2, k_layer.norm2)
        transfer_norm_weights(pt_layer.norm3, k_layer.norm3)

        # Cross Attention
        pt_ca = pt_layer.cross_attn
        k_ca = k_layer.cross_attn
        transfer_linear_weights(pt_ca.value_proj, k_ca.value_proj)
        transfer_linear_weights(pt_ca.output_proj, k_ca.output_proj)
        transfer_linear_weights(pt_ca.sampling_offsets, k_ca.sampling_offsets)
        transfer_linear_weights(pt_ca.attention_weights, k_ca.attention_weights)

        # FFN
        transfer_linear_weights(pt_layer.linear1, k_layer.linear1)
        transfer_linear_weights(pt_layer.linear2, k_layer.linear2)


def load_pretrained_weights_into_pytorch(pt_model, weights_path):
    if not os.path.exists(weights_path):
        pytest.skip(f"Weights file not found at: {weights_path}")

    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    renamed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("transformer."):
            new_key = k.replace("transformer.", "")
            renamed_state_dict[new_key] = v
        elif "enc_out_class_embed" in k or "enc_out_bbox_embed" in k:
            renamed_state_dict[k] = v

    model_state_dict = pt_model.state_dict()
    final_state_dict = {}

    for k, v in renamed_state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                final_state_dict[k] = v

    pt_model.load_state_dict(final_state_dict, strict=False)
    print("Weights loaded successfully.")


# =============================================================================
# 2. CONFIGURATION & SETUP
# =============================================================================


def get_config():
    return {
        "d_model": 256,
        "sa_nhead": 8,
        "ca_nhead": 8,
        "num_queries": 300,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "num_feature_levels": 4,
        "dec_n_points": 1,
        "two_stage": True,
        "bbox_reparam": True,
        "group_detr": 1,
        "num_classes": 91,
        "batch_size": 2,
    }


def build_pytorch_model(config, weights_path):
    model = Transformer_PyTorch(
        d_model=config["d_model"],
        sa_nhead=config["sa_nhead"],
        ca_nhead=config["ca_nhead"],
        num_queries=config["num_queries"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        two_stage=config["two_stage"],
        num_feature_levels=config["num_feature_levels"],
        dec_n_points=config["dec_n_points"],
        bbox_reparam=config["bbox_reparam"],
        group_detr=config["group_detr"],
        return_intermediate_dec=True,
    )
    model.enc_out_class_embed = nn.ModuleList(
        [
            nn.Linear(config["d_model"], config["num_classes"])
            for _ in range(config["group_detr"])
        ]
    )
    model.enc_out_bbox_embed = nn.ModuleList(
        [
            MLP_PyTorch(config["d_model"], config["d_model"], 4, 3)
            for _ in range(config["group_detr"])
        ]
    )
    load_pretrained_weights_into_pytorch(model, weights_path)
    if not hasattr(model.decoder, "bbox_embed") or model.decoder.bbox_embed is None:
        model.decoder.bbox_embed = model.enc_out_bbox_embed[0]
    model.eval()
    return model


def build_keras_model(config):
    enc_class = [
        layers.Dense(config["num_classes"]) for _ in range(config["group_detr"])
    ]
    enc_bbox = [
        MLP_Keras(config["d_model"], config["d_model"], 4, 3)
        for _ in range(config["group_detr"])
    ]

    for l in enc_class:
        l.build((None, config["d_model"]))
    for m in enc_bbox:
        m.build((None, config["d_model"]))

    return Transformer_Keras(
        d_model=config["d_model"],
        sa_nhead=config["sa_nhead"],
        ca_nhead=config["ca_nhead"],
        num_queries=config["num_queries"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        two_stage=config["two_stage"],
        num_feature_levels=config["num_feature_levels"],
        dec_n_points=config["dec_n_points"],
        bbox_reparam=config["bbox_reparam"],
        group_detr=config["group_detr"],
        return_intermediate_dec=True,
        enc_out_class_embed=enc_class,
        enc_out_bbox_embed=enc_bbox,
    )


def transfer_all_weights(pt_model, k_model, config):
    print("Transferring weights...")
    for i in range(config["num_decoder_layers"]):
        transfer_decoder_layer_weights(
            pt_model.decoder.layers[i],
            k_model.decoder.decoder_layers[i],
            config["sa_nhead"],
        )
    transfer_norm_weights(pt_model.decoder.norm, k_model.decoder.norm)
    transfer_mlp_weights(
        pt_model.decoder.ref_point_head, k_model.decoder.ref_point_head
    )

    if config["two_stage"]:
        for i in range(config["group_detr"]):
            transfer_linear_weights(pt_model.enc_output[i], k_model.enc_output[i])
            transfer_norm_weights(
                pt_model.enc_output_norm[i], k_model.enc_output_norm[i]
            )
            transfer_linear_weights(
                pt_model.enc_out_class_embed[i], k_model.enc_out_class_embed[i]
            )
            transfer_mlp_weights(
                pt_model.enc_out_bbox_embed[i], k_model.enc_out_bbox_embed[i]
            )

    k_model.decoder.bbox_embed = k_model.enc_out_bbox_embed[0]


# =============================================================================
# 3. DATA GENERATION
# =============================================================================


def generate_random_numpy_data(config):

    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    srcs, masks, pos_embeds = [], [], []
    for h, w in spatial_shapes:
        srcs.append(
            np.random.randn(config["batch_size"], config["d_model"], h, w).astype(
                "float32"
            )
        )
        m = np.zeros((config["batch_size"], h, w), dtype="bool")
        m[:, -2:, :] = True
        masks.append(m)
        pos_embeds.append(
            np.random.randn(config["batch_size"], config["d_model"], h, w).astype(
                "float32"
            )
        )
    query_feat = np.random.randn(
        config["batch_size"], config["num_queries"], config["d_model"]
    ).astype("float32")
    ref_embed = np.random.randn(config["batch_size"], config["num_queries"], 4).astype(
        "float32"
    )
    return srcs, masks, pos_embeds, query_feat, ref_embed


# =============================================================================
# 4. GRANULAR LAYER STEPS (THE PARITY CHECK LOGIC)
# =============================================================================


def _get_pytorch_ref_points(pt_model, ref_unsig, valid_ratios_pt, config):
    """
    Generates reference points and query position embeddings for PyTorch.
    CRITICAL FIX: Uses gen_sineembed_for_position_pt (Torch version) to avoid JAX/Torch conflicts.
    """
    if not pt_model.decoder.bbox_reparam:
        ref = ref_unsig.sigmoid()
    else:
        ref = ref_unsig

    obj_center = ref[..., :4]
    ref_input = (
        obj_center[:, :, None]
        * torch.cat([valid_ratios_pt, valid_ratios_pt], -1)[:, None]
    )

    # Use the imported PyTorch specific utility
    q_sine = gen_sineembed_for_position_pt(
        ref_input[:, :, 0, :], config["d_model"] // 2
    )
    q_pos = pt_model.decoder.ref_point_head(q_sine)

    return obj_center, ref_input, q_pos, q_sine


def step_self_attn(i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, q_pos_pt, q_pos_np):
    """Executes Self-Attention and parity check."""
    # PyTorch
    q_pt = k_pt = curr_tgt_pt + q_pos_pt
    sa_out_pt = pt_layer.self_attn(q_pt, k_pt, curr_tgt_pt, need_weights=False)[0]

    # Keras
    sa_out_k = k_layer.self_attn(
        query=curr_tgt_np + q_pos_np, key=curr_tgt_np + q_pos_np, value=curr_tgt_np
    )

    check_parity(i, "Self-Attn", sa_out_pt, sa_out_k)
    return sa_out_pt, sa_out_k


def step_norm1(i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, sa_out_pt, sa_out_k):
    """Executes Norm1 (Add & Norm). Returns output and Teacher-Forced Keras input."""
    norm1_pt = pt_layer.norm1(curr_tgt_pt + sa_out_pt)
    norm1_k = k_layer.norm1(curr_tgt_np + sa_out_k)
    check_parity(i, "Norm 1", norm1_pt, norm1_k)

    # Teacher Forcing: Keras next step uses PyTorch output
    return norm1_pt, norm1_pt.detach().numpy()


def step_cross_attn(
    i,
    pt_layer,
    k_layer,
    curr_tgt_pt,
    curr_tgt_np,
    q_pos_pt,
    q_pos_np,
    ref_input_pt,
    ref_input_np,
    mem_pt,
    mem_np,
    shapes_pt,
    shapes_np,
    lvl_start_pt,
    lvl_start_np,
    mask_pt,
    mask_np_list,
    src_flat_list,
    shapes_list,
):
    """Executes Cross-Attention."""
    # PyTorch
    q_ca_pt = curr_tgt_pt + q_pos_pt
    ca_out_pt = pt_layer.cross_attn(
        q_ca_pt, ref_input_pt, mem_pt, shapes_pt, lvl_start_pt, mask_pt
    )

    # Keras
    q_ca_np = curr_tgt_np + q_pos_np
    ca_out_k = k_layer.cross_attn(
        q_ca_np,
        ref_input_np,
        mem_np,
        shapes_np,
        lvl_start_np,
        input_padding_mask=np.concatenate(mask_np_list, axis=1),
        input_flatten_list=src_flat_list,
        input_padding_mask_list=mask_np_list,
        input_spatial_shapes_list=shapes_list,
    )

    check_parity(i, "Cross-Attn", ca_out_pt, ca_out_k)
    return ca_out_pt, ca_out_k


def step_norm2(i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, ca_out_pt, ca_out_k):
    """Executes Norm2."""
    norm2_pt = pt_layer.norm2(curr_tgt_pt + ca_out_pt)
    norm2_k = k_layer.norm2(curr_tgt_np + ca_out_k)
    check_parity(i, "Norm 2", norm2_pt, norm2_k)
    return norm2_pt, norm2_pt.detach().numpy()


def step_ffn_block(i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np):
    """Executes FFN (Linear1 -> Act -> Linear2)."""
    # PyTorch
    ffn_inner_pt = pt_layer.activation(pt_layer.linear1(curr_tgt_pt))
    ffn_out_pt = pt_layer.linear2(ffn_inner_pt)

    # Keras
    ffn_inner_k = k_layer.activation(k_layer.linear1(curr_tgt_np))
    ffn_out_k = k_layer.linear2(ffn_inner_k)

    check_parity(i, "FFN", ffn_out_pt, ffn_out_k)
    return ffn_out_pt, ffn_out_k


def step_norm3(i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, ffn_out_pt, ffn_out_k):
    """Executes Norm3."""
    norm3_pt = pt_layer.norm3(curr_tgt_pt + ffn_out_pt)
    norm3_k = k_layer.norm3(curr_tgt_np + ffn_out_k)
    check_parity(i, "Norm 3", norm3_pt, norm3_k)
    return norm3_pt, norm3_pt.detach().numpy()


def step_box_refine(
    i, pt_model, k_model, curr_tgt_pt, curr_tgt_np, ref_unsig_pt, ref_unsig_np
):
    """Executes Bbox refinement delta and updates reference points."""
    if pt_model.decoder.lite_refpoint_refine:
        return ref_unsig_pt, ref_unsig_np

    delta_pt = pt_model.decoder.bbox_embed(curr_tgt_pt)
    delta_k = k_model.decoder.bbox_embed(curr_tgt_np)

    check_parity(i, "Box Delta", delta_pt, delta_k)

    new_ref_pt = pt_model.decoder.refpoints_refine(ref_unsig_pt, delta_pt)
    return new_ref_pt.detach(), new_ref_pt.detach().numpy()


def compute_valid_ratios(masks):
    """Computes valid height/width ratios for masks."""
    ratios = []
    for m in masks:
        H, W = m.shape[1], m.shape[2]
        valid_H = np.sum(~m[:, :, 0], axis=1)
        valid_W = np.sum(~m[:, 0, :], axis=1)
        ratios.append(np.stack([valid_W / W, valid_H / H], axis=-1))
    return np.stack(ratios, axis=1).astype("float32")


def prepare_pytorch_tensors(srcs, masks, pos_embeds, valid_ratios_np):
    """Converts numpy data to flattened PyTorch tensors AND returns flattened numpy data."""
    src_flat, mask_flat, pos_flat = [], [], []
    spatial_shapes = []

    for src, mask, pos in zip(srcs, masks, pos_embeds):
        bs, c, h, w = src.shape
        spatial_shapes.append((h, w))
        # Logic calculated once here
        src_flat.append(src.reshape(bs, c, -1).swapaxes(1, 2))
        pos_flat.append(pos.reshape(bs, c, -1).swapaxes(1, 2))
        mask_flat.append(mask.reshape(bs, -1))

    # Create Numpy Memory
    mem_np = np.concatenate(src_flat, axis=1)

    # Create PyTorch Tensors from the Numpy data
    memory = torch.from_numpy(mem_np)
    mask_padded = torch.from_numpy(np.concatenate(mask_flat, axis=1))
    lvl_pos = torch.from_numpy(np.concatenate(pos_flat, axis=1))

    shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long)
    level_start = torch.cat(
        (shapes_tensor.new_zeros((1,)), shapes_tensor.prod(1).cumsum(0)[:-1])
    )
    valid_ratios = torch.from_numpy(valid_ratios_np)

    return (
        memory,
        mask_padded,
        lvl_pos,
        shapes_tensor,
        level_start,
        valid_ratios,
        spatial_shapes,
        src_flat,
        mask_flat,
        mem_np,
    )


def setup_parity_test_env():
    """Encapsulates the shared setup logic for all parity tests."""
    print("Setting up test environment...")
    np.random.seed(42)
    torch.manual_seed(42)
    weights_path = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\rf-detr-base-coco.pth"
    config = get_config()

    pt_model = build_pytorch_model(config, weights_path)
    k_model = build_keras_model(config)

    # Generate Data
    srcs, masks, pos_embeds, query_feat, ref_embed = generate_random_numpy_data(config)
    valid_ratios = compute_valid_ratios(masks)

    # Keras Dummy Pass (Initialize shapes)
    _ = k_model(
        srcs=srcs,
        masks=masks,
        pos_embeds=pos_embeds,
        refpoint_embed=ref_embed[0],
        query_feat=query_feat[0],
        training=False,
    )

    # Transfer Weights
    transfer_all_weights(pt_model, k_model, config)

    return {
        "config": config,
        "pt_model": pt_model,
        "k_model": k_model,
        "data": (srcs, masks, pos_embeds, query_feat, ref_embed),
        "valid_ratios": valid_ratios,
    }


# =============================================================================
# 4. Pytest Fixtures
# =============================================================================


@pytest.fixture
def config_params():
    """Returns fixed model configuration parameters."""
    return {
        "d_model": 256,
        "sa_nhead": 8,
        "ca_nhead": 8,
        "num_queries": 20,
        "num_decoder_layers": 2,
        "dim_feedforward": 512,
        "num_feature_levels": 3,
        "dec_n_points": 4,
        "return_intermediate": True,
        "batch_size": 2,
        "num_classes": 10,
    }


@pytest.fixture
def generated_inputs(config_params):
    """Generates consistent random inputs for testing."""
    d_model = config_params["d_model"]
    num_queries = config_params["num_queries"]
    batch_size = config_params["batch_size"]

    # Deterministic seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    spatial_shapes = [(8, 8), (4, 4), (2, 2)]
    srcs_np = []
    masks_np = []
    pos_embeds_np = []

    for h, w in spatial_shapes:
        srcs_np.append(np.random.randn(batch_size, d_model, h, w).astype("float32"))
        mask = np.zeros((batch_size, h, w), dtype="bool")
        mask[:, -1, :] = True
        mask[:, :, -1] = True
        masks_np.append(mask)
        pos_embeds_np.append(
            np.random.randn(batch_size, d_model, h, w).astype("float32")
        )

    # Unbatched for PyTorch expansion compatibility
    query_feat_np = np.random.randn(num_queries, d_model).astype("float32")
    refpoint_embed_np = np.random.randn(num_queries, 4).astype("float32")

    # PyTorch Inputs
    srcs_pt = [torch.from_numpy(x) for x in srcs_np]
    masks_pt = [torch.from_numpy(x) for x in masks_np]
    pos_embeds_pt = [torch.from_numpy(x) for x in pos_embeds_np]
    query_feat_pt = torch.from_numpy(query_feat_np)
    refpoint_embed_pt = torch.from_numpy(refpoint_embed_np)

    return {
        "np": (srcs_np, masks_np, pos_embeds_np, query_feat_np, refpoint_embed_np),
        "pt": (srcs_pt, masks_pt, pos_embeds_pt, query_feat_pt, refpoint_embed_pt),
    }


# =============================================================================
# 5. MAIN TEST FUNCTION WITH PARAMETRIZATION and random weights.
# =============================================================================
def test_mlp_equivalence():
    # --- Configuration ---
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    num_layers = 3
    batch_size = 8

    # --- Instantiate Models ---
    pt_model = MLP_PyTorch(input_dim, hidden_dim, output_dim, num_layers)
    k_model = MLP_Keras(input_dim, hidden_dim, output_dim, num_layers)

    # --- Create Dummy Data (Random Weights) ---
    # We generate weights in PyTorch format (out_features, in_features)
    # Layer 1: Input -> Hidden
    w1 = np.random.randn(hidden_dim, input_dim).astype(np.float32)
    b1 = np.random.randn(hidden_dim).astype(np.float32)

    # Layer 2: Hidden -> Hidden
    w2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
    b2 = np.random.randn(hidden_dim).astype(np.float32)

    # Layer 3: Hidden -> Output
    w3 = np.random.randn(output_dim, hidden_dim).astype(np.float32)
    b3 = np.random.randn(output_dim).astype(np.float32)

    weights = [(w1, b1), (w2, b2), (w3, b3)]

    # --- Load Weights into PyTorch Model ---
    with torch.no_grad():
        # Layer 0
        pt_model.layers[0].weight.copy_(torch.from_numpy(weights[0][0]))
        pt_model.layers[0].bias.copy_(torch.from_numpy(weights[0][1]))
        # Layer 1
        pt_model.layers[1].weight.copy_(torch.from_numpy(weights[1][0]))
        pt_model.layers[1].bias.copy_(torch.from_numpy(weights[1][1]))
        # Layer 2
        pt_model.layers[2].weight.copy_(torch.from_numpy(weights[2][0]))
        pt_model.layers[2].bias.copy_(torch.from_numpy(weights[2][1]))

    # --- Load Weights into Keras Model (With Transpose!) ---
    # Keras Dense weights are [kernel, bias]
    # Kernel shape must be (input_dim, units), so we transpose the PyTorch weights

    # Layer 0
    k_model.mlp_layers.layers[0].set_weights([weights[0][0].T, weights[0][1]])
    # Layer 1
    k_model.mlp_layers.layers[1].set_weights([weights[1][0].T, weights[1][1]])
    # Layer 2
    k_model.mlp_layers.layers[2].set_weights([weights[2][0].T, weights[2][1]])

    # --- Run Forward Pass ---
    # Generate random input
    x_np = np.random.randn(batch_size, input_dim).astype(np.float32)

    # 1. PyTorch Output
    pt_model.eval()  # Set to eval mode (though no dropout/batchnorm here, good practice)
    pt_out_tensor = pt_model(torch.from_numpy(x_np))
    pt_out = pt_out_tensor.detach().numpy()

    # 2. Keras Output
    k_out_tensor = k_model(x_np)
    # Convert Keras tensor to numpy (works for TF, JAX, or Torch backend)
    k_out = np.array(k_out_tensor)

    # --- Compare ---
    print(f"\nPyTorch Output Sample:\n{pt_out[0]}")
    print(f"Keras Output Sample:\n{k_out[0]}")

    # Assert equality within a very small tolerance (float32 precision)
    np.testing.assert_allclose(
        pt_out, k_out, rtol=1e-4, atol=1e-5, err_msg="Outputs do not match!"
    )

    print(
        "\nSUCCESS: The Keras implementation matches the PyTorch implementation exactly."
    )


def test_ms_deform_attn_equivalence():
    # --- Configuration ---
    N = 2  # Batch size
    d_model = 32  # Hidden dimension
    n_levels = 3  # Number of feature levels
    n_heads = 4  # Number of attention heads
    n_points = 4  # Number of sampling points
    Len_q = 5  # Query length

    # Define spatial shapes for the levels (H, W)
    spatial_shapes_list = [(8, 8), (4, 4), (2, 2)]

    # Calculate derived dimensions
    total_len_in = sum(h * w for h, w in spatial_shapes_list)

    # Prepare Spatial Shapes and Start Indices
    spatial_shapes_np = np.array(spatial_shapes_list, dtype="int32")
    level_start_index_np = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes_list])[:-1]]
    )
    level_start_index_np = level_start_index_np.astype("int32")

    # Torch (for PyTorch model)
    spatial_shapes_pt = torch.from_numpy(spatial_shapes_np).long()
    level_start_index_pt = torch.from_numpy(level_start_index_np).long()

    # --- Instantiate Models ---
    pt_model = MSDeformAttn_PyTorch(d_model, n_levels, n_heads, n_points)
    k_model = MSDeformAttn_Keras(d_model, n_levels, n_heads, n_points)

    # --- Initialize Keras Model ---
    dummy_q = np.zeros((1, Len_q, d_model), dtype="float32")
    dummy_ref = np.zeros((1, Len_q, n_levels, 4), dtype="float32")  # 4D ref points
    dummy_in = np.zeros((1, total_len_in, d_model), dtype="float32")

    # Run one pass to initialize variables
    _ = k_model(dummy_q, dummy_ref, dummy_in, spatial_shapes_np, level_start_index_np)

    with torch.no_grad():
        # 1. Value Projection
        k_model.value_proj.set_weights(
            [
                pt_model.value_proj.weight.detach().numpy().T,
                pt_model.value_proj.bias.detach().numpy(),
            ]
        )

        # 2. Output Projection
        k_model.output_proj.set_weights(
            [
                pt_model.output_proj.weight.detach().numpy().T,
                pt_model.output_proj.bias.detach().numpy(),
            ]
        )

        # 3. Sampling Offsets
        k_model.sampling_offsets.set_weights(
            [
                pt_model.sampling_offsets.weight.detach().numpy().T,
                pt_model.sampling_offsets.bias.detach().numpy(),
            ]
        )

        # 4. Attention Weights
        k_model.attention_weights.set_weights(
            [
                pt_model.attention_weights.weight.detach().numpy().T,
                pt_model.attention_weights.bias.detach().numpy(),
            ]
        )

    print("Weights transferred successfully.")

    # --- Generate Random Inputs ---
    # Query
    query_np = np.random.randn(N, Len_q, d_model).astype("float32")

    # Reference Points
    ref_points_np = np.random.rand(N, Len_q, n_levels, 4).astype("float32")

    # Input Flatten (Memory)
    input_flatten_np = np.random.randn(N, total_len_in, d_model).astype("float32")

    # --- Run Forward Pass ---

    # 1. PyTorch Output
    pt_model.eval()
    pt_out_tensor = pt_model(
        query=torch.from_numpy(query_np),
        reference_points=torch.from_numpy(ref_points_np),
        input_flatten=torch.from_numpy(input_flatten_np),
        input_spatial_shapes=spatial_shapes_pt,
        input_level_start_index=level_start_index_pt,
        input_padding_mask=None,
    )
    pt_out = pt_out_tensor.detach().numpy()

    # 2. Keras Output
    k_out_tensor = k_model(
        query=query_np,
        reference_points=ref_points_np,
        input_flatten=input_flatten_np,
        input_spatial_shapes=spatial_shapes_np,
        input_level_start_index=level_start_index_np,
        input_padding_mask=None,
    )
    k_out = np.array(k_out_tensor)

    # --- Compare ---
    print(f"\nPyTorch Output Mean: {np.mean(pt_out):.6f}")
    print(f"Keras Output Mean:   {np.mean(k_out):.6f}")

    diff = np.abs(pt_out - k_out)
    print(f"Max Absolute Difference: {np.max(diff):.8f}")

    np.testing.assert_allclose(
        pt_out,
        k_out,
        rtol=1e-4,
        atol=1e-5,
        err_msg="MSDeformAttn Outputs do not match!",
    )

    print("\nSUCCESS: The Keras MSDeformAttn implementation matches PyTorch.")


def test_decoder_layer_equivalence():
    # --- Configuration ---
    d_model = 256
    sa_nhead = 8
    ca_nhead = 8
    dim_feedforward = 1024
    dropout = 0.0  # Disable dropout for deterministic testing
    activation = "relu"
    num_feature_levels = 4
    dec_n_points = 4

    batch_size = 2
    num_queries = 10

    # Total memory length must match sum of spatial_shapes areas
    num_memory = 100

    # --- Instantiate Models ---
    pt_model = TransformerDecoderLayer_PyTorch(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
    )
    pt_model.eval()

    k_model = TransformerDecoderLayer_Keras(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
    )

    # --- Prepare Dummy Inputs & Build Keras Model ---
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype("float32")
    memory_np = np.random.randn(batch_size, num_memory, d_model).astype("float32")
    query_pos_np = np.random.randn(batch_size, num_queries, d_model).astype("float32")

    # Reference points: (bs, nq, n_levels, 2)
    ref_points_np = np.random.rand(
        batch_size, num_queries, num_feature_levels, 2
    ).astype("float32")
    spatial_shapes_list = [(5, 5), (5, 5), (5, 5), (5, 5)]
    spatial_shapes_np = np.array(spatial_shapes_list, dtype="int32")

    level_start_index_np = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes_list])[:-1]]
    ).astype("int32")

    # Run a dummy pass to build Keras model weights
    k_model(
        tgt=tgt_np,
        memory=memory_np,
        training=False,
        query_pos=query_pos_np,
        reference_points=ref_points_np,
        spatial_shapes=spatial_shapes_np,
        level_start_index=level_start_index_np,
    )

    # --- Weight Transfer (PyTorch -> Keras) ---

    with torch.no_grad():
        # 1. Self Attention (MultiHeadAttention)
        pt_sa = pt_model.self_attn

        # Retrieve PyTorch weights (Concatenated Q, K, V)
        qkv_w = pt_sa.in_proj_weight.detach().numpy()  # (3*d_model, d_model)
        qkv_b = pt_sa.in_proj_bias.detach().numpy()  # (3*d_model,)

        # Split into Q, K, V
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)

        # Retrieve Output Projection
        out_w = pt_sa.out_proj.weight.detach().numpy()  # (d_model, d_model)
        out_b = pt_sa.out_proj.bias.detach().numpy()  # (d_model,)

        # --- Helper Functions for Reshaping ---

        def to_keras_mha_input_kernel(w, n_head):
            """Reshapes PyTorch (Out, In) -> Keras (In, Heads, Head_Dim)"""
            w = w.T  # Transpose to (In, Out) -> (256, 256)
            d_model = w.shape[0]
            head_dim = w.shape[1] // n_head
            # Reshape to (256, 8, 32)
            return w.reshape(d_model, n_head, head_dim)

        def to_keras_mha_bias(b, n_head):
            """Reshapes PyTorch (Out,) -> Keras (Heads, Head_Dim)"""
            head_dim = b.shape[0] // n_head
            # Reshape to (8, 32)
            return b.reshape(n_head, head_dim)

        def to_keras_mha_output_kernel(w, n_head):
            """Reshapes PyTorch OutProj (Out, In) -> Keras (Heads, Head_Dim, Out)"""
            w = w.T

            d_model_out = w.shape[1]
            head_dim = w.shape[0] // n_head
            return w.reshape(n_head, head_dim, d_model_out)

        # --- Apply Reshaping ---
        q_w_k = to_keras_mha_input_kernel(q_w, sa_nhead)
        k_w_k = to_keras_mha_input_kernel(k_w, sa_nhead)
        v_w_k = to_keras_mha_input_kernel(v_w, sa_nhead)

        q_b_k = to_keras_mha_bias(q_b, sa_nhead)
        k_b_k = to_keras_mha_bias(k_b, sa_nhead)
        v_b_k = to_keras_mha_bias(v_b, sa_nhead)

        out_w_k = to_keras_mha_output_kernel(out_w, sa_nhead)
        out_b_k = out_b  # Bias matches (256,)

        # Set Weights
        k_model.self_attn.set_weights(
            [q_w_k, q_b_k, k_w_k, k_b_k, v_w_k, v_b_k, out_w_k, out_b_k]
        )

        # 2. Norms
        k_model.norm1.set_weights(
            [
                pt_model.norm1.weight.detach().numpy(),
                pt_model.norm1.bias.detach().numpy(),
            ]
        )
        k_model.norm2.set_weights(
            [
                pt_model.norm2.weight.detach().numpy(),
                pt_model.norm2.bias.detach().numpy(),
            ]
        )
        k_model.norm3.set_weights(
            [
                pt_model.norm3.weight.detach().numpy(),
                pt_model.norm3.bias.detach().numpy(),
            ]
        )

        # 3. Cross Attention (MSDeformAttn)
        pt_ca = pt_model.cross_attn
        k_ca = k_model.cross_attn

        k_ca.value_proj.set_weights(
            [
                pt_ca.value_proj.weight.detach().numpy().T,
                pt_ca.value_proj.bias.detach().numpy(),
            ]
        )
        k_ca.output_proj.set_weights(
            [
                pt_ca.output_proj.weight.detach().numpy().T,
                pt_ca.output_proj.bias.detach().numpy(),
            ]
        )
        k_ca.sampling_offsets.set_weights(
            [
                pt_ca.sampling_offsets.weight.detach().numpy().T,
                pt_ca.sampling_offsets.bias.detach().numpy(),
            ]
        )
        k_ca.attention_weights.set_weights(
            [
                pt_ca.attention_weights.weight.detach().numpy().T,
                pt_ca.attention_weights.bias.detach().numpy(),
            ]
        )

        # 4. Feed Forward Network
        k_model.linear1.set_weights(
            [
                pt_model.linear1.weight.detach().numpy().T,
                pt_model.linear1.bias.detach().numpy(),
            ]
        )
        k_model.linear2.set_weights(
            [
                pt_model.linear2.weight.detach().numpy().T,
                pt_model.linear2.bias.detach().numpy(),
            ]
        )

    print("Weights transferred successfully.")

    # --- Run Comparison ---

    # Prepare Inputs (Convert to Tensor for PyTorch)
    tgt_pt = torch.from_numpy(tgt_np)
    memory_pt = torch.from_numpy(memory_np)
    query_pos_pt = torch.from_numpy(query_pos_np)
    ref_points_pt = torch.from_numpy(ref_points_np)
    spatial_shapes_pt = torch.from_numpy(spatial_shapes_np).long()
    level_start_index_pt = torch.from_numpy(level_start_index_np).long()

    # 1. PyTorch Forward
    pt_out = (
        pt_model(
            tgt=tgt_pt,
            memory=memory_pt,
            pos=None,
            query_pos=query_pos_pt,
            reference_points=ref_points_pt,
            spatial_shapes=spatial_shapes_pt,
            level_start_index=level_start_index_pt,
        )
        .detach()
        .numpy()
    )

    # 2. Keras Forward
    k_out = k_model(
        tgt=tgt_np,
        memory=memory_np,
        training=False,
        pos=None,
        query_pos=query_pos_np,
        reference_points=ref_points_np,
        spatial_shapes=spatial_shapes_np,
        level_start_index=level_start_index_np,
    )
    k_out = np.array(k_out)

    # --- Assertions ---
    print(f"\nPyTorch Output Mean: {np.mean(pt_out):.6f}")
    print(f"Keras Output Mean:   {np.mean(k_out):.6f}")

    diff = np.abs(pt_out - k_out)
    max_diff = np.max(diff)
    print(f"Max Absolute Difference: {max_diff:.8f}")

    np.testing.assert_allclose(
        pt_out,
        k_out,
        rtol=1e-4,
        atol=1e-5,
        err_msg="TransformerDecoderLayer outputs do not match!",
    )

    print("\nSUCCESS: The Keras TransformerDecoderLayer matches PyTorch.")


def test_transformer_decoder_equivalence():
    # --- Configuration ---
    d_model = 256
    num_layers = 3
    sa_nhead = 8
    ca_nhead = 8
    dim_feedforward = 1024
    dropout = 0.0
    activation = "relu"
    num_feature_levels = 3
    dec_n_points = 4
    return_intermediate = True
    lite_refpoint_refine = False
    bbox_reparam = False

    batch_size = 2
    num_queries = 10

    # Define spatial shapes for memory
    spatial_shapes_list = [(8, 8), (4, 4), (2, 2)]
    total_len_in = sum(h * w for h, w in spatial_shapes_list)

    # --- Prepare Inputs ---
    tgt_np = np.random.randn(batch_size, num_queries, d_model).astype("float32")
    memory_np = np.random.randn(batch_size, total_len_in, d_model).astype("float32")
    refpoints_unsigmoid_np = np.random.randn(batch_size, num_queries, 4).astype(
        "float32"
    )

    spatial_shapes_np = np.array(spatial_shapes_list, dtype="int32")
    level_start_index_np = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes_list])[:-1]]
    ).astype("int32")

    valid_ratios_np = (
        np.ones((batch_size, num_feature_levels, 2), dtype="float32") * 0.99
    )

    tgt_pt = torch.from_numpy(tgt_np)
    memory_pt = torch.from_numpy(memory_np)
    refpoints_unsigmoid_pt = torch.from_numpy(refpoints_unsigmoid_np)
    spatial_shapes_pt = torch.from_numpy(spatial_shapes_np).long()
    level_start_index_pt = torch.from_numpy(level_start_index_np).long()
    valid_ratios_pt = torch.from_numpy(valid_ratios_np)

    # --- Instantiate Layer Models (Templates) ---
    pt_layer = TransformerDecoderLayer_PyTorch(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
    )
    k_layer = TransformerDecoderLayer_Keras(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        skip_self_attn=False,
    )

    # --- Assign bbox_embed weights BEFORE model creation ---
    if not lite_refpoint_refine:
        bbox_embed_pt = MLP_PyTorch(d_model, d_model, 4, 3)
        pt_layer.bbox_embed = bbox_embed_pt

        bbox_embed_k = MLP_Keras(d_model, d_model, 4, 3)
        k_layer.bbox_embed = bbox_embed_k

        # Build Keras bbox_embed to init variables
        bbox_embed_k.build((None, d_model))

    # --- Instantiate Decoder Models ---
    pt_model = TransformerDecoder_PyTorch(
        pt_layer,
        num_layers,
        norm=torch.nn.LayerNorm(d_model),
        return_intermediate=return_intermediate,
        d_model=d_model,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
    )
    if not lite_refpoint_refine:
        pt_model.bbox_embed = bbox_embed_pt

    k_model = TransformerDecoder_Keras(
        k_layer,
        num_layers,
        norm=keras.layers.LayerNormalization(epsilon=1e-5),
        return_intermediate=return_intermediate,
        d_model=d_model,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
    )
    pt_model.eval()

    # --- Build Keras Model (Dummy Call) ---
    _ = k_model(
        tgt=tgt_np,
        memory=memory_np,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=refpoints_unsigmoid_np,
        level_start_index=level_start_index_np,
        spatial_shapes=spatial_shapes_np,
        valid_ratios=valid_ratios_np,
        training=False,
    )

    # --- Weight Transfer (PyTorch -> Keras) ---

    # 1. Transfer MLP (ref_point_head)
    transfer_mlp_weights(pt_model.ref_point_head, k_model.ref_point_head)

    # 2. Transfer Decoder Layers (cloned modules)
    for i in range(num_layers):
        pt_dec_layer = pt_model.layers[i]
        k_dec_layer = k_model.decoder_layers[i]

        transfer_decoder_layer_weights(
            pt_dec_layer, k_dec_layer, d_model, sa_nhead, ca_nhead
        )

        # 3. Transfer Layer Norm (if present)
        if k_model.norm is not None:
            k_model.norm.set_weights(
                [
                    pt_model.norm.weight.detach().numpy(),
                    pt_model.norm.bias.detach().numpy(),
                ]
            )

        # 4. Transfer bbox_embed (Shared or per-layer?)
        if not lite_refpoint_refine and i == 0:
            transfer_mlp_weights(pt_model.bbox_embed, k_model.bbox_embed)

    print("Weights transferred successfully.")

    # --- Run Forward Pass ---

    # 1. PyTorch Output
    with torch.no_grad():
        pt_out_tuple = pt_model(
            tgt=tgt_pt,
            memory=memory_pt,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            refpoints_unsigmoid=refpoints_unsigmoid_pt,
            level_start_index=level_start_index_pt,
            spatial_shapes=spatial_shapes_pt,
            valid_ratios=valid_ratios_pt,
        )
        pt_hs = pt_out_tuple[0].detach().numpy()
        pt_ref = pt_out_tuple[1].detach().numpy()

    # 2. Keras Output
    k_out_tuple = k_model(
        tgt=tgt_np,
        memory=memory_np,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=refpoints_unsigmoid_np,
        level_start_index=level_start_index_np,
        spatial_shapes=spatial_shapes_np,
        valid_ratios=valid_ratios_np,
        training=False,
    )
    k_hs = np.array(k_out_tuple[0])
    k_ref = np.array(k_out_tuple[1])

    # --- Compare ---
    print(f"\nComparing Hidden States (hs):")
    diff_hs = np.abs(pt_hs - k_hs)
    max_diff_hs = np.max(diff_hs)
    print(f"Max Absolute Difference (hs): {max_diff_hs:.8f}")

    np.testing.assert_allclose(
        pt_hs,
        k_hs,
        rtol=1e-4,
        atol=1e-5,
        err_msg="TransformerDecoder Hidden States (hs) do not match!",
    )

    print(f"\nComparing Reference Points (ref):")
    diff_ref = np.abs(pt_ref - k_ref)
    max_diff_ref = np.max(diff_ref)
    print(f"Max Absolute Difference (ref): {max_diff_ref:.8f}")

    np.testing.assert_allclose(
        pt_ref,
        k_ref,
        rtol=1e-4,
        atol=1e-5,
        err_msg="TransformerDecoder Reference Points (ref) do not match!",
    )

    print("\nFINAL SUCCESS: The Keras TransformerDecoder matches PyTorch.")


@pytest.mark.parametrize(
    "two_stage, bbox_reparam, group_detr",
    [
        (True, False, 1),  # Standard Two-Stage
        (True, True, 1),  # BBox Reparam
        (False, False, 1),  # One-Stage
        (True, False, 2),  # Group DETR
    ],
    ids=["Standard Two-Stage", "BBox Reparam", "One-Stage", "Group DETR"],
)
def test_transformer_decoder_parity(
    config_params, generated_inputs, two_stage, bbox_reparam, group_detr
):
    """
    Main test function verifying parity between PyTorch and Keras DINO implementations.
    """
    # Unpack config
    d_model = config_params["d_model"]
    sa_nhead = config_params["sa_nhead"]
    ca_nhead = config_params["ca_nhead"]
    num_queries = config_params["num_queries"]
    num_decoder_layers = config_params["num_decoder_layers"]
    dim_feedforward = config_params["dim_feedforward"]
    num_feature_levels = config_params["num_feature_levels"]
    dec_n_points = config_params["dec_n_points"]
    return_intermediate = config_params["return_intermediate"]
    num_classes = config_params["num_classes"]

    # Unpack inputs
    srcs_np, masks_np, pos_embeds_np, query_feat_np, refpoint_embed_np = (
        generated_inputs["np"]
    )
    srcs_pt, masks_pt, pos_embeds_pt, query_feat_pt, refpoint_embed_pt = (
        generated_inputs["pt"]
    )

    # --- 1. Instantiate Models ---

    # PyTorch Model
    pt_model = Transformer_PyTorch(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        two_stage=two_stage,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
        return_intermediate_dec=return_intermediate,
    )
    pt_model.eval()

    # Define external heads for PyTorch
    enc_out_class_embed_pt = nn.ModuleList(
        [nn.Linear(d_model, num_classes) for _ in range(group_detr)]
    )
    enc_out_bbox_embed_pt = nn.ModuleList(
        [MLP_PyTorch(d_model, d_model, 4, 3) for _ in range(group_detr)]
    )
    pt_model.enc_out_class_embed = enc_out_class_embed_pt
    pt_model.enc_out_bbox_embed = enc_out_bbox_embed_pt

    # Keras Model Heads
    enc_out_class_embed_k = []
    enc_out_bbox_embed_k = []
    for _ in range(group_detr):
        l = layers.Dense(num_classes)
        l.build((None, d_model))
        enc_out_class_embed_k.append(l)
        m = MLP_Keras(d_model, d_model, 4, 3)
        m.build((None, d_model))
        enc_out_bbox_embed_k.append(m)

    # Keras Model
    k_model = Transformer_Keras(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        two_stage=two_stage,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
        return_intermediate_dec=return_intermediate,
        enc_out_class_embed=enc_out_class_embed_k,
        enc_out_bbox_embed=enc_out_bbox_embed_k,
    )

    # --- 2. Build Keras Model (Dummy Call) ---
    dummy_query_feat = query_feat_np
    dummy_refpoint_embed = refpoint_embed_np

    if group_detr > 1:
        dummy_query_feat = np.tile(query_feat_np, (group_detr, 1))
        dummy_refpoint_embed = np.tile(refpoint_embed_np, (group_detr, 1))

    _ = k_model(
        srcs=srcs_np,
        masks=masks_np,
        pos_embeds=pos_embeds_np,
        refpoint_embed=dummy_refpoint_embed,
        query_feat=dummy_query_feat,
        training=True,  # Always True here to build all paths
    )

    # --- 3. Weight Transfer ---
    for i in range(num_decoder_layers):
        transfer_decoder_layer_weights(
            pt_model.decoder.layers[i], k_model.decoder.decoder_layers[i], sa_nhead
        )

    if k_model.decoder.norm is not None:
        transfer_norm_weights(pt_model.decoder.norm, k_model.decoder.norm)

    transfer_mlp_weights(
        pt_model.decoder.ref_point_head, k_model.decoder.ref_point_head
    )

    for i in range(group_detr):
        transfer_linear_weights(
            pt_model.enc_out_class_embed[i], k_model.enc_out_class_embed[i]
        )
        transfer_mlp_weights(
            pt_model.enc_out_bbox_embed[i], k_model.enc_out_bbox_embed[i]
        )

        if two_stage:
            transfer_linear_weights(pt_model.enc_output[i], k_model.enc_output[i])
            transfer_norm_weights(
                pt_model.enc_output_norm[i], k_model.enc_output_norm[i]
            )

    pt_bbox_embed = getattr(pt_model.decoder, "bbox_embed", None)
    if not pt_model.decoder.lite_refpoint_refine and pt_bbox_embed is None:
        pt_model.decoder.bbox_embed = enc_out_bbox_embed_pt[0]
        if k_model.decoder.bbox_embed is None:
            k_model.decoder.bbox_embed = enc_out_bbox_embed_k[0]

    # --- 4. Run Comparisons ---
    with torch.no_grad():
        pt_outs = pt_model(
            srcs=srcs_pt,
            masks=masks_pt,
            pos_embeds=pos_embeds_pt,
            refpoint_embed=refpoint_embed_pt,
            query_feat=query_feat_pt,
        )
        # Unpack based on two_stage
        if two_stage:
            pt_hs, pt_ref, pt_mem_ts, pt_box_ts = pt_outs
            pt_mem_ts = pt_mem_ts.detach().numpy()
            pt_box_ts = pt_box_ts.detach().numpy()
        else:
            pt_hs, pt_ref, _, _ = pt_outs
            pt_mem_ts, pt_box_ts = None, None

        pt_hs = pt_hs.detach().numpy()
        pt_ref = pt_ref.detach().numpy()

    # Inference mode comparison
    k_outs = k_model(
        srcs=srcs_np,
        masks=masks_np,
        pos_embeds=pos_embeds_np,
        refpoint_embed=refpoint_embed_np,
        query_feat=query_feat_np,
        training=False,
    )
    k_hs = np.array(k_outs[0])
    k_ref = np.array(k_outs[1])

    if two_stage:
        k_mem_ts = np.array(k_outs[2])
        k_box_ts = np.array(k_outs[3])
    else:
        k_mem_ts, k_box_ts = None, None

    # --- 5. Assertions ---
    np.testing.assert_allclose(
        pt_hs, k_hs, rtol=1e-4, atol=1e-4, err_msg="Hidden States Mismatch"
    )

    np.testing.assert_allclose(
        pt_ref, k_ref, rtol=1e-4, atol=1e-4, err_msg="Reference Points Mismatch"
    )

    if two_stage:
        np.testing.assert_allclose(
            pt_mem_ts, k_mem_ts, rtol=1e-4, atol=1e-4, err_msg="TS Memory Mismatch"
        )
        np.testing.assert_allclose(
            pt_box_ts, k_box_ts, rtol=1e-4, atol=1e-4, err_msg="TS Boxes Mismatch"
        )


# =============================================================================
# 6. MAIN TEST FUNCTION with pretrained weights
# =============================================================================


def test_granular_block_by_block_parity():
    print(f"\n{'='*20} GRANULAR BLOCK-BY-BLOCK DEBUGGING {'='*20}")

    # 1. Unified Setup
    env = setup_parity_test_env()
    config = env["config"]
    pt_model = env["pt_model"]
    k_model = env["k_model"]
    srcs, masks, pos_embeds, query_feat, ref_embed = env["data"]
    valid_ratios_np = env["valid_ratios"]

    # 2. Prepare Tensors (Now returns both PT and NP data)
    (
        mem_pt,
        mask_pt,
        lvl_pos_pt,
        shapes_pt,
        lvl_start_pt,
        valid_ratios_pt,
        shapes_list,
        # Unpack the numpy versions here directly
        src_flat_list,
        mask_flat_list,
        mem_np,
    ) = prepare_pytorch_tensors(srcs, masks, pos_embeds, valid_ratios_np)

    # 3. Init Loop Variables
    curr_tgt_pt = torch.from_numpy(query_feat)
    curr_tgt_np = query_feat
    ref_unsig_pt = torch.from_numpy(ref_embed)
    ref_unsig_np = ref_embed

    print(
        f"\n{'Layer':<5} | {'Block':<15} | {'MAE':<12} | {'Max Diff':<12} | {'Status'}"
    )
    print("-" * 65)

    # 4. Layer Loop
    for i in range(config["num_decoder_layers"]):
        pt_layer = pt_model.decoder.layers[i]
        k_layer = k_model.decoder.decoder_layers[i]

        # A. Ref Points (Fixes the TypeError)
        _, ref_in_pt, q_pos_pt, _ = _get_pytorch_ref_points(
            pt_model, ref_unsig_pt, valid_ratios_pt, config
        )
        q_pos_np = q_pos_pt.detach().numpy()
        ref_in_np = ref_in_pt.detach().numpy()

        # B. Self Attn
        sa_pt, sa_k = step_self_attn(
            i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, q_pos_pt, q_pos_np
        )

        # C. Norm 1 + Teacher Forcing
        curr_tgt_pt, curr_tgt_np = step_norm1(
            i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, sa_pt, sa_k
        )

        # D. Cross Attn
        ca_pt, ca_k = step_cross_attn(
            i,
            pt_layer,
            k_layer,
            curr_tgt_pt,
            curr_tgt_np,
            q_pos_pt,
            q_pos_np,
            ref_in_pt,
            ref_in_np,
            mem_pt,
            mem_np,
            shapes_pt,
            shapes_pt.numpy(),
            lvl_start_pt,
            lvl_start_pt.numpy(),
            mask_pt,
            mask_flat_list,
            src_flat_list,
            shapes_list,
        )

        # E. Norm 2 + Teacher Forcing
        curr_tgt_pt, curr_tgt_np = step_norm2(
            i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, ca_pt, ca_k
        )

        # F. FFN
        ffn_pt, ffn_k = step_ffn_block(i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np)

        # G. Norm 3 + Teacher Forcing
        curr_tgt_pt, curr_tgt_np = step_norm3(
            i, pt_layer, k_layer, curr_tgt_pt, curr_tgt_np, ffn_pt, ffn_k
        )

        # H. Box Refine
        ref_unsig_pt, ref_unsig_np = step_box_refine(
            i, pt_model, k_model, curr_tgt_pt, curr_tgt_np, ref_unsig_pt, ref_unsig_np
        )


def test_transformer_output_parity():
    print(f"\n{'='*20} FULL TRANSFORMER END-TO-END PARITY {'='*20}")
    # 1. Unified Setup
    env = setup_parity_test_env()
    pt_model = env["pt_model"]
    k_model = env["k_model"]
    srcs_np, masks_np, pos_embeds_np, query_feat_np, ref_embed_np = env["data"]

    # PyTorch Inference
    def to_pt_list(x_list):
        return [torch.from_numpy(x) for x in x_list]

    srcs_pt = to_pt_list(srcs_np)
    masks_pt = to_pt_list(masks_np)
    pos_embeds_pt = to_pt_list(pos_embeds_np)

    # PyTorch model expects unbatched shared embeddings (nq, dim)
    query_feat_pt = torch.from_numpy(query_feat_np[0])
    ref_embed_pt = torch.from_numpy(ref_embed_np[0])

    print("Running PyTorch inference...")
    with torch.no_grad():
        pt_outs = pt_model(
            srcs_pt, masks_pt, pos_embeds_pt, ref_embed_pt, query_feat_pt
        )

    # Keras Inference
    print("Running Keras inference...")
    k_outs = k_model(
        srcs=srcs_np,
        masks=masks_np,
        pos_embeds=pos_embeds_np,
        refpoint_embed=ref_embed_np[0],
        query_feat=query_feat_np[0],
        training=False,
    )

    # Comparison
    output_names = [
        "Hidden States",
        "Reference Points",
        "Enc Output Memory",
        "Enc Output Boxes",
    ]
    print(f"\n{'Output Name':<20} | {'MAE':<12} | {'Max Diff':<12} | {'Status'}")
    print("-" * 65)

    for i, name in enumerate(output_names):
        pt_val = pt_outs[i]
        k_val = k_outs[i]

        if pt_val is None or k_val is None:
            status = (
                "PASS (None)"
                if pt_val is None and k_val is None
                else "FAIL (None Mismatch)"
            )
            mae, max_diff = 0.0, 0.0
        else:
            pt_val_np = pt_val.cpu().numpy()
            k_val_np = np.array(k_val)
            diff = np.abs(pt_val_np - k_val_np)
            mae = np.mean(diff)
            max_diff = np.max(diff)
            status = "PASS" if mae < 1e-4 else "FAIL"

        print(f"{name:<20} | {mae:.8f}     | {max_diff:.8f}     | {status}")


if __name__ == "__main__":
    test_granular_block_by_block_parity()
    test_transformer_output_parity()
    pytest.main(["-v", __file__])
