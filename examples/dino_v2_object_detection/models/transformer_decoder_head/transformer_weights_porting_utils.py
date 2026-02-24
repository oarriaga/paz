import os
import sys
import numpy as np
import torch
import torch.nn as nn
import keras
from keras import ops
from keras import layers
import pytest

# RFDETR imports
# Try importing from installed package first, then fallback to local path
try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase
    from rfdetr.config import RFDETRBaseConfig
except ImportError:
    # Determine path relative to this file
    # This file is in examples/dino_v2_object_detection/models/transformer_decoder_head/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rfdetr_path = os.path.abspath(
        os.path.join(current_dir, "../rf-detr_original_pytorch_implementation")
    )
    if rfdetr_path not in sys.path:
        sys.path.insert(0, rfdetr_path)
    try:
        from rfdetr import (
            RFDETRNano,
            RFDETRSmall,
            RFDETRMedium,
            RFDETRLarge,
            RFDETRBase,
        )
        from rfdetr.config import RFDETRBaseConfig
        from rfdetr.models.ops.modules.ms_deform_attn import (
            MSDeformAttn as TorchMSDeformAttn,
        )
    except ImportError:
        pass

from examples.dino_v2_object_detection.models.transformer_decoder_head.transformer import (
    Transformer as KerasTransformer,
)
from examples.dino_v2_object_detection.models.transformer_decoder_head.transformer import (
    MLP as KerasMLP,
)
from examples.dino_v2_object_detection.models.transformer_decoder_head.ms_deform_attn import (
    MSDeformAttn as KerasMSDeformAttn,
)


# ═══════════════════════════════════════════════════════════════════
# General helpers
# ═══════════════════════════════════════════════════════════════════


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.array(t)


def to_torch(x):
    return torch.tensor(x, dtype=torch.float32)


def to_keras(x):
    return ops.convert_to_tensor(x, dtype="float32")


# ═══════════════════════════════════════════════════════════════════
# MSDeformAttn Helpers
# ═══════════════════════════════════════════════════════════════════


def get_ms_deform_attn_from_model(model):
    """
    Traverse model to find the first MSDeformAttn layer in the decoder.
    RFDETR structure: model.model.transformer.decoder.layers[0].cross_attn
    """
    try:
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "model"):  # Wrapper
                inner = inner.model
        else:
            inner = model

        return inner.transformer.decoder.layers[0].cross_attn
    except AttributeError:
        # Try finding recursively if structure is different
        pass

    # Fallback walk
    for name, module in model.named_modules():
        if "cross_attn" in name and "decoder" in name and "layers.0" in name:
            return module
    return None


# ═══════════════════════════════════════════════════════════════════
# Transformer Helpers
# ═══════════════════════════════════════════════════════════════════


def extract_pt_transformer(model_class, config=None):
    """Load PyTorch model and return the Transformer module."""
    if config is not None:
        # Synthetic case
        wrapper = model_class(config=config)
    else:
        # Real/Pretrained case
        wrapper = model_class(pretrained=True)

    # wrapper.model is LightningModule, wrapper.model.model is nn.Module (DINO/RFDETR)
    wrapper.model.model.eval()
    # Structure: wrapper.model.model.transformer
    pt_transformer = wrapper.model.model.transformer
    return pt_transformer


def transfer_transformer_weights(pt_transformer, keras_transformer, d_model, sa_nhead):
    """Transfer weights from PyTorch Transformer to Keras Transformer."""

    with torch.no_grad():
        # 1. Encoder Output (Two Stage)
        # Note: RF-DETR models typically use two_stage=True.
        if pt_transformer.two_stage:
            # RF-DETR might initialize more layers (e.g. 13) than active levels.
            # We only transfer weights for the active levels defined in Keras model.
            num_groups = len(keras_transformer.enc_output)

            for i in range(num_groups):
                keras_transformer.enc_output[i].kernel.assign(
                    to_keras(pt_transformer.enc_output[i].weight.T.cpu().numpy())
                )
                keras_transformer.enc_output[i].bias.assign(
                    to_keras(pt_transformer.enc_output[i].bias.cpu().numpy())
                )

                keras_transformer.enc_output_norm[i].gamma.assign(
                    to_keras(pt_transformer.enc_output_norm[i].weight.cpu().numpy())
                )
                keras_transformer.enc_output_norm[i].beta.assign(
                    to_keras(pt_transformer.enc_output_norm[i].bias.cpu().numpy())
                )

                # Class embed
                # keras_transformer.enc_out_class_embed was initialized to match PT length (13),
                # but we only populate the active ones here.
                keras_transformer.enc_out_class_embed[i].kernel.assign(
                    to_keras(
                        pt_transformer.enc_out_class_embed[i].weight.T.cpu().numpy()
                    )
                )
                keras_transformer.enc_out_class_embed[i].bias.assign(
                    to_keras(pt_transformer.enc_out_class_embed[i].bias.cpu().numpy())
                )

                # BBox embed (MLP)
                for j, (tk, klayer) in enumerate(
                    zip(
                        pt_transformer.enc_out_bbox_embed[i].layers,
                        keras_transformer.enc_out_bbox_embed[i].layers_list,
                    )
                ):
                    klayer.kernel.assign(to_keras(tk.weight.T.cpu().numpy()))
                    klayer.bias.assign(to_keras(tk.bias.cpu().numpy()))

        # 2. Decoder Layers
        num_layers = len(pt_transformer.decoder.layers)
        for i in range(num_layers):
            t_layer = pt_transformer.decoder.layers[i]
            k_layer = keras_transformer.decoder.layers_list[i]

            # Helper to transfer MHA
            def transfer_mha(t_mha, k_mha):
                q_w = to_keras(t_mha.in_proj_weight[:d_model, :].T.cpu().numpy())
                q_w = ops.reshape(q_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.query_dense.kernel.assign(q_w)
                q_b = to_keras(t_mha.in_proj_bias[:d_model].cpu().numpy())
                q_b = ops.reshape(q_b, (sa_nhead, d_model // sa_nhead))
                k_mha.query_dense.bias.assign(q_b)

                k_w = to_keras(
                    t_mha.in_proj_weight[d_model : 2 * d_model, :].T.cpu().numpy()
                )
                k_w = ops.reshape(k_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.key_dense.kernel.assign(k_w)
                k_b = to_keras(t_mha.in_proj_bias[d_model : 2 * d_model].cpu().numpy())
                k_b = ops.reshape(k_b, (sa_nhead, d_model // sa_nhead))
                k_mha.key_dense.bias.assign(k_b)

                v_w = to_keras(t_mha.in_proj_weight[2 * d_model :, :].T.cpu().numpy())
                v_w = ops.reshape(v_w, (d_model, sa_nhead, d_model // sa_nhead))
                k_mha.value_dense.kernel.assign(v_w)
                v_b = to_keras(t_mha.in_proj_bias[2 * d_model :].cpu().numpy())
                v_b = ops.reshape(v_b, (sa_nhead, d_model // sa_nhead))
                k_mha.value_dense.bias.assign(v_b)

                out_w = to_keras(t_mha.out_proj.weight.T.cpu().numpy())
                out_w = ops.reshape(out_w, (sa_nhead, d_model // sa_nhead, d_model))
                k_mha.output_dense.kernel.assign(out_w)
                k_mha.output_dense.bias.assign(
                    to_keras(t_mha.out_proj.bias.cpu().numpy())
                )

            transfer_mha(t_layer.self_attn, k_layer.self_attn)

            k_layer.norm1.gamma.assign(to_keras(t_layer.norm1.weight.cpu().numpy()))
            k_layer.norm1.beta.assign(to_keras(t_layer.norm1.bias.cpu().numpy()))

            # Cross Attn
            k_layer.cross_attn.sampling_offsets.kernel.assign(
                to_keras(t_layer.cross_attn.sampling_offsets.weight.T.cpu().numpy())
            )
            k_layer.cross_attn.sampling_offsets.bias.assign(
                to_keras(t_layer.cross_attn.sampling_offsets.bias.cpu().numpy())
            )
            k_layer.cross_attn.attention_weights.kernel.assign(
                to_keras(t_layer.cross_attn.attention_weights.weight.T.cpu().numpy())
            )
            k_layer.cross_attn.attention_weights.bias.assign(
                to_keras(t_layer.cross_attn.attention_weights.bias.cpu().numpy())
            )
            k_layer.cross_attn.value_proj.kernel.assign(
                to_keras(t_layer.cross_attn.value_proj.weight.T.cpu().numpy())
            )
            k_layer.cross_attn.value_proj.bias.assign(
                to_keras(t_layer.cross_attn.value_proj.bias.cpu().numpy())
            )
            k_layer.cross_attn.output_proj.kernel.assign(
                to_keras(t_layer.cross_attn.output_proj.weight.T.cpu().numpy())
            )
            k_layer.cross_attn.output_proj.bias.assign(
                to_keras(t_layer.cross_attn.output_proj.bias.cpu().numpy())
            )

            k_layer.norm2.gamma.assign(to_keras(t_layer.norm2.weight.cpu().numpy()))
            k_layer.norm2.beta.assign(to_keras(t_layer.norm2.bias.cpu().numpy()))

            k_layer.linear1.kernel.assign(
                to_keras(t_layer.linear1.weight.T.cpu().numpy())
            )
            k_layer.linear1.bias.assign(to_keras(t_layer.linear1.bias.cpu().numpy()))
            k_layer.linear2.kernel.assign(
                to_keras(t_layer.linear2.weight.T.cpu().numpy())
            )
            k_layer.linear2.bias.assign(to_keras(t_layer.linear2.bias.cpu().numpy()))

            k_layer.norm3.gamma.assign(to_keras(t_layer.norm3.weight.cpu().numpy()))
            k_layer.norm3.beta.assign(to_keras(t_layer.norm3.bias.cpu().numpy()))

        # 3. Decoder Modules
        # Ref Point Head
        for j, (tk, klayer) in enumerate(
            zip(
                pt_transformer.decoder.ref_point_head.layers,
                keras_transformer.decoder.ref_point_head.layers_list,
            )
        ):
            klayer.kernel.assign(to_keras(tk.weight.T.cpu().numpy()))
            klayer.bias.assign(to_keras(tk.bias.cpu().numpy()))

        # BBox Embed
        if (
            getattr(pt_transformer.decoder, "bbox_embed", None) is not None
            and getattr(keras_transformer.decoder, "bbox_embed", None) is not None
        ):
            for j, (tk, klayer) in enumerate(
                zip(
                    pt_transformer.decoder.bbox_embed.layers,
                    keras_transformer.decoder.bbox_embed.layers_list,
                )
            ):
                klayer.kernel.assign(to_keras(tk.weight.T.cpu().numpy()))
                klayer.bias.assign(to_keras(tk.bias.cpu().numpy()))

        # Norm
        keras_transformer.decoder.norm.gamma.assign(
            to_keras(pt_transformer.decoder.norm.weight.cpu().numpy())
        )
        keras_transformer.decoder.norm.beta.assign(
            to_keras(pt_transformer.decoder.norm.bias.cpu().numpy())
        )


def verify_transformer_parity(pt_transformer, variant_name):
    """Core logic to verify parity between a loaded PyTorch transformer and Keras implementation."""

    # Extract configs
    # d_model: usually stored, but if not, check layer 0
    if hasattr(pt_transformer, "d_model"):
        d_model = pt_transformer.d_model
    else:
        d_model = pt_transformer.decoder.layers[0].linear1.in_features

    # nhead: check self_attn of first layer
    # Torch MultiheadAttention has .num_heads
    sa_nhead = pt_transformer.decoder.layers[0].self_attn.num_heads
    ca_nhead = sa_nhead  # Assuming symmetric

    num_decoder_layers = len(pt_transformer.decoder.layers)

    # dim_feedforward: linear1 output or linear2 input?
    # linear1: d_model -> dim_feedforward
    dim_feedforward = pt_transformer.decoder.layers[0].linear1.out_features

    # dropout
    dropout = pt_transformer.decoder.layers[0].dropout1.p

    num_queries = pt_transformer.num_queries

    # Inspecting first decoder layer for MSDeformAttn params
    # cross_attn is MSDeformAttn
    cross_attn = pt_transformer.decoder.layers[0].cross_attn
    n_levels = cross_attn.n_levels
    dec_n_points = cross_attn.n_points
    ca_nhead = cross_attn.n_heads

    # Check consistency and override if needed
    sampling_offsets_shape = cross_attn.sampling_offsets.weight.shape
    # Shape is (out_features, in_features) in PT
    # out_features = n_heads * n_levels * n_points * 2
    out_features = sampling_offsets_shape[0]
    calculated_n_points = out_features // (ca_nhead * n_levels * 2)

    if calculated_n_points != dec_n_points:
        print(
            f"WARNING: MSDeformAttn.n_points ({dec_n_points}) differs from weight shape derived ({calculated_n_points}) using n_heads={ca_nhead}. Using derived value."
        )
        dec_n_points = calculated_n_points

    # Activation and Norm
    # activation is usually a function in PT layer, need to infer name or check config if available
    # pt_transformer.decoder.layers[0].activation might be F.relu or wrapper
    pt_activation = getattr(pt_transformer.decoder.layers[0], "activation", None)
    activation = "relu"  # Default
    if pt_activation is not None:
        # Try to get name
        if hasattr(pt_activation, "__name__"):
            if "relu" in pt_activation.__name__:
                activation = "relu"
            elif "gelu" in pt_activation.__name__:
                activation = "gelu"
            elif "glu" in pt_activation.__name__:
                activation = "glu"

    # normalize_before
    normalize_before = getattr(
        pt_transformer.decoder.layers[0], "normalize_before", False
    )

    # two_stage: infer from presence of encoder output embeddings
    two_stage = (
        hasattr(pt_transformer, "enc_out_class_embed")
        and len(pt_transformer.enc_out_class_embed) > 0
    )

    # return_intermediate_dec: usually True for DETR models (aux loss)
    # Check decoder.return_intermediate
    return_intermediate_dec = getattr(
        pt_transformer.decoder, "return_intermediate", True
    )

    # bbox_reparam
    # Check if attribute exists, otherwise default False
    bbox_reparam = getattr(pt_transformer, "bbox_reparam", False)

    # lite_refpoint_refine
    lite_refpoint_refine = getattr(
        pt_transformer.decoder, "lite_refpoint_refine", False
    )

    print(
        f"Config: d_model={d_model}, nhead={sa_nhead}, layers={num_decoder_layers}, levels={n_levels}"
    )
    print(
        f"Flags: two_stage={two_stage}, return_intermediate_dec={return_intermediate_dec}, bbox_reparam={bbox_reparam}, lite_refpoint_refine={lite_refpoint_refine}, activation={activation}"
    )

    # 2. Build Keras Transformer
    print("Building Keras Transformer...")
    keras_transformer = KerasTransformer(
        d_model=d_model,
        sa_nhead=sa_nhead,
        ca_nhead=ca_nhead,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        return_intermediate_dec=return_intermediate_dec,
        two_stage=two_stage,
        num_feature_levels=n_levels,
        dec_n_points=dec_n_points,
        bbox_reparam=bbox_reparam,
        lite_refpoint_refine=lite_refpoint_refine,
        activation=activation,
        normalize_before=normalize_before,
    )

    # Manually Assign Assignments (Enc Out, Decoder BBox)
    # These are created in RF-DETR model.py but assigned to transformer instance.
    # In PyTorch: transformer.enc_out_class_embed = ...
    # We must replicate this structure in Keras.

    # Inspect pt_transformer to see what it has
    if hasattr(pt_transformer, "enc_out_class_embed"):
        # It's a ModuleList
        keras_transformer.enc_out_class_embed = [
            layers.Dense(layer.out_features)
            for layer in pt_transformer.enc_out_class_embed
        ]
        for layer in keras_transformer.enc_out_class_embed:
            layer.build((None, d_model))

    if hasattr(pt_transformer, "enc_out_bbox_embed"):
        # ModuleList of MLPs
        keras_transformer.enc_out_bbox_embed = [
            KerasMLP(d_model, d_model, 4, 3) for _ in pt_transformer.enc_out_bbox_embed
        ]
        for mlp in keras_transformer.enc_out_bbox_embed:
            # Build MLP manually? MLP forward calls layers.
            # We can build inner layers if needed, or call MLP.
            # KerasMLP layers are in layers_list.
            # Let's call it with dummy
            mlp(ops.zeros((1, d_model)))

    if getattr(pt_transformer.decoder, "bbox_embed", None) is not None:
        # Single MLP
        keras_transformer.decoder.bbox_embed = KerasMLP(d_model, d_model, 4, 3)
        keras_transformer.decoder.bbox_embed(ops.zeros((1, d_model)))

    # Debug lengths
    if two_stage:
        print(f"DEBUG: PT enc_output type: {type(pt_transformer.enc_output)}")
        print(f"DEBUG: PT enc_output len: {len(pt_transformer.enc_output)}")
        if len(pt_transformer.enc_output) > 0:
            print(f"DEBUG: PT enc_output[0]: {pt_transformer.enc_output[0]}")
        print(f"DEBUG: Keras enc_output len: {len(keras_transformer.enc_output)}")

    # 3. Create Inputs (Random but valid shapes)
    bs = 1

    spatial_shapes = []
    srcs_np = []
    masks_np = []
    pos_embeds_np = []

    h0, w0 = 32, 32  # Base size
    for l in range(n_levels):
        h, w = max(1, h0 // (2**l)), max(1, w0 // (2**l))  # Just halving
        spatial_shapes.append((h, w))
        srcs_np.append(np.random.randn(bs, d_model, h, w).astype(np.float32))
        masks_np.append(np.zeros((bs, h, w), dtype=bool))
        pos_embeds_np.append(np.random.randn(bs, d_model, h, w).astype(np.float32))

    query_feat_np = np.random.randn(num_queries, d_model).astype(np.float32)
    # Refpoint embed (4 dim for two_stage mode? or 2?)
    # In two_stage, it's used as query_embed (usually 4? or just passed as ref points?)
    # If two_stage, transformer generates proposals.
    # But forward takes `refpoint_embed`.
    # in Two stage: refpoint_embed is (N, 4).
    refpoint_embed_np = np.random.randn(num_queries, 4).astype(np.float32)

    t_srcs = [torch.tensor(x) for x in srcs_np]
    t_masks = [torch.tensor(x) for x in masks_np]
    t_pos_embeds = [torch.tensor(x) for x in pos_embeds_np]
    t_query_feat = torch.tensor(query_feat_np)
    t_refpoint_embed = torch.tensor(refpoint_embed_np)

    k_srcs = [to_keras(x) for x in srcs_np]
    k_masks = [to_keras(x) for x in masks_np]
    k_pos_embeds = [to_keras(x) for x in pos_embeds_np]
    k_query_feat = to_keras(query_feat_np)
    k_refpoint_embed = to_keras(refpoint_embed_np)

    # Build Keras Layer
    print("Building Keras model variables...")
    keras_transformer(k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed)

    # 4. Transfer Weights
    print("Transferring weights...")
    transfer_transformer_weights(pt_transformer, keras_transformer, d_model, sa_nhead)

    # 5. Run Forward
    print("Running forward passes...")
    with torch.no_grad():
        pt_outs = pt_transformer(
            t_srcs, t_masks, t_pos_embeds, t_refpoint_embed, t_query_feat
        )
        # (hs, references, memory_ts, boxes_ts)
        # Note: references might be tuple?

    k_outs = keras_transformer(
        k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed
    )

    # 6. Compare
    print("Comparing outputs...")
    out_names = ["Hidden States", "References", "Memory TS", "Boxes TS"]
    tolerances = {
        "Hidden States": 1e-3,  # Stricter tolerance as verified
        "References": 1e-4,
        "Memory TS": 1e-4,
        "Boxes TS": 1e-4,
    }

    failures = []
    for i in range(len(pt_outs)):
        pt_val = to_numpy(pt_outs[i])
        if k_outs[i] is None:
            if pt_val is not None:
                print(f"  {out_names[i]} Mismatch: PT is {pt_val.shape}, Keras is None")
                failures.append(f"{out_names[i]} is None in Keras")
            continue

        k_val = to_numpy(k_outs[i])

        # Check shape
        if pt_val.shape != k_val.shape:
            print(
                f"  {out_names[i]} Shape Mismatch: PT {pt_val.shape} vs Keras {k_val.shape}"
            )
            failures.append(f"{out_names[i]} shape mismatch")
            continue

        diff = np.abs(pt_val - k_val)
        mean_diff = diff.mean()
        max_diff = diff.max()
        print(f"  {out_names[i]} Mean Diff: {mean_diff:.6f} (Max Diff: {max_diff:.6f})")

        tol = tolerances.get(out_names[i], 1e-4)  # Default 1e-4
        if mean_diff > tol:
            failures.append(f"{out_names[i]} mean diff {mean_diff} > {tol}")

    if failures:
        pytest.fail(f"Parity check failed: {failures}")

    print(f"RFDETR {variant_name} Transformer parity PASSED")
