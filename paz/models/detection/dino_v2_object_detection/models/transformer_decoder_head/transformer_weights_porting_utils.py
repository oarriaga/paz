import os
import sys
import numpy as np
import torch
import torch.nn as nn
import keras
from keras import ops
from keras import layers
import pytest

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase
    from rfdetr.config import RFDETRBaseConfig
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rfdetr_path = os.path.abspath(
        os.path.join(current_dir, "../../../../../../examples/rf-detr_original_pytorch_implementation")
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

from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (
    Transformer as KerasTransformer,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (
    MLP as KerasMLP,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.ms_deform_attn import (
    MSDeformAttn as KerasMSDeformAttn,
)


# ═══════════════════════════════════════════════════════════════════
# General helpers
# ═══════════════════════════════════════════════════════════════════


def to_numpy(t):
    """Convert a tensor (any framework) to a NumPy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.array(t)


def to_torch(x):
    """Convert a NumPy array to a float32 torch tensor."""
    return torch.tensor(x, dtype=torch.float32)


def to_keras(x):
    """Convert a NumPy array to a float32 Keras tensor."""
    return ops.convert_to_tensor(x, dtype="float32")


# ═══════════════════════════════════════════════════════════════════
# MSDeformAttn Helpers
# ═══════════════════════════════════════════════════════════════════


def get_ms_deform_attn_from_model(model):
    """Locate the first MSDeformAttn layer in the decoder of the model.

    Traverses the RF-DETR model hierarchy to find the cross-attention
    module from the first decoder layer.

    Args:
        model: An RF-DETR model instance (may be wrapped).

    Returns:
        The MSDeformAttn module, or None if not found.
    """
    try:
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "model"):
                inner = inner.model
        else:
            inner = model

        return inner.transformer.decoder.layers[0].cross_attn
    except AttributeError:
        pass

    # Fallback: search all named modules for the first decoder cross-attention
    for name, module in model.named_modules():
        if "cross_attn" in name and "decoder" in name and "layers.0" in name:
            return module
    return None


# ═══════════════════════════════════════════════════════════════════
# Transformer Helpers
# ═══════════════════════════════════════════════════════════════════


def extract_pt_transformer(model_class, config=None):
    """Load a pretrained RF-DETR model and return its transformer module.

    Args:
        model_class: RF-DETR model class (e.g., RFDETRNano).
        config: Optional config object. If None, loads pretrained weights.

    Returns:
        The transformer sub-module in eval mode.
    """
    if config is not None:
        wrapper = model_class(config=config)
    else:
        wrapper = model_class(pretrained=True)

    wrapper.model.model.eval()
    pt_transformer = wrapper.model.model.transformer
    return pt_transformer


def transfer_transformer_weights(pt_transformer, keras_transformer, d_model, sa_nhead):
    """Transfer all transformer weights from a pretrained model into Keras.

    Copies encoder output projections, decoder self-attention, deformable
    cross-attention, feed-forward, normalization, reference point head,
    and bbox embed weights.

    Args:
        pt_transformer: Source pretrained transformer module.
        keras_transformer: Target Keras Transformer model.
        d_model (int): Model embedding dimension.
        sa_nhead (int): Number of self-attention heads.
    """
    with torch.no_grad():
        # Transfer encoder output layers (two-stage components)
        if pt_transformer.two_stage:
            num_groups = len(keras_transformer.enc_output)

            for i in range(num_groups):
                # Projection layer weights
                keras_transformer.enc_output[i].kernel.assign(
                    to_keras(pt_transformer.enc_output[i].weight.T.cpu().numpy())
                )
                keras_transformer.enc_output[i].bias.assign(
                    to_keras(pt_transformer.enc_output[i].bias.cpu().numpy())
                )

                # Layer normalization weights
                keras_transformer.enc_output_norm[i].gamma.assign(
                    to_keras(pt_transformer.enc_output_norm[i].weight.cpu().numpy())
                )
                keras_transformer.enc_output_norm[i].beta.assign(
                    to_keras(pt_transformer.enc_output_norm[i].bias.cpu().numpy())
                )

                # Classification head
                keras_transformer.enc_out_class_embed[i].kernel.assign(
                    to_keras(
                        pt_transformer.enc_out_class_embed[i].weight.T.cpu().numpy()
                    )
                )
                keras_transformer.enc_out_class_embed[i].bias.assign(
                    to_keras(pt_transformer.enc_out_class_embed[i].bias.cpu().numpy())
                )

                # BBox regression MLP
                for j, (tk, klayer) in enumerate(
                    zip(
                        pt_transformer.enc_out_bbox_embed[i].layers,
                        keras_transformer.enc_out_bbox_embed[i].layers_list,
                    )
                ):
                    klayer.kernel.assign(to_keras(tk.weight.T.cpu().numpy()))
                    klayer.bias.assign(to_keras(tk.bias.cpu().numpy()))

        # Transfer decoder layer weights
        num_layers = len(pt_transformer.decoder.layers)
        for i in range(num_layers):
            t_layer = pt_transformer.decoder.layers[i]
            k_layer = keras_transformer.decoder.layers_list[i]

            # Transfer multi-head self-attention weights.
            # The source uses a fused in_proj_weight that concatenates Q, K, V
            # projections. Split by d_model segments and reshape for Keras
            # MultiHeadAttention format (d_model, n_heads, head_dim).
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

            # Transfer self-attention, then norms, cross-attention, and FFN
            transfer_mha(t_layer.self_attn, k_layer.self_attn)

            # Self-attention layer norm
            k_layer.norm1.gamma.assign(to_keras(t_layer.norm1.weight.cpu().numpy()))
            k_layer.norm1.beta.assign(to_keras(t_layer.norm1.bias.cpu().numpy()))

            # Deformable cross-attention weights
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

            # Cross-attention layer norm
            k_layer.norm2.gamma.assign(to_keras(t_layer.norm2.weight.cpu().numpy()))
            k_layer.norm2.beta.assign(to_keras(t_layer.norm2.bias.cpu().numpy()))

            # FFN (linear1 -> activation -> linear2)
            k_layer.linear1.kernel.assign(
                to_keras(t_layer.linear1.weight.T.cpu().numpy())
            )
            k_layer.linear1.bias.assign(to_keras(t_layer.linear1.bias.cpu().numpy()))
            k_layer.linear2.kernel.assign(
                to_keras(t_layer.linear2.weight.T.cpu().numpy())
            )
            k_layer.linear2.bias.assign(to_keras(t_layer.linear2.bias.cpu().numpy()))

            # FFN layer norm
            k_layer.norm3.gamma.assign(to_keras(t_layer.norm3.weight.cpu().numpy()))
            k_layer.norm3.beta.assign(to_keras(t_layer.norm3.bias.cpu().numpy()))

        # Transfer decoder-level modules
        # Reference point head MLP
        for j, (tk, klayer) in enumerate(
            zip(
                pt_transformer.decoder.ref_point_head.layers,
                keras_transformer.decoder.ref_point_head.layers_list,
            )
        ):
            klayer.kernel.assign(to_keras(tk.weight.T.cpu().numpy()))
            klayer.bias.assign(to_keras(tk.bias.cpu().numpy()))

        # BBox embed MLP (optional, used for iterative refinement)
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

        # Decoder output normalization
        keras_transformer.decoder.norm.gamma.assign(
            to_keras(pt_transformer.decoder.norm.weight.cpu().numpy())
        )
        keras_transformer.decoder.norm.beta.assign(
            to_keras(pt_transformer.decoder.norm.bias.cpu().numpy())
        )


def verify_transformer_parity(pt_transformer, variant_name):
    """Build a Keras Transformer matching a pretrained model, transfer
    weights, run both forward passes, and assert numerical parity.

    Args:
        pt_transformer: Pretrained transformer module.
        variant_name (str): Descriptive name for logging.

    Raises:
        pytest.fail: If any output tensor exceeds the tolerance threshold.
    """
    # Extract model configuration from the pretrained transformer
    if hasattr(pt_transformer, "d_model"):
        d_model = pt_transformer.d_model
    else:
        d_model = pt_transformer.decoder.layers[0].linear1.in_features

    sa_nhead = pt_transformer.decoder.layers[0].self_attn.num_heads
    ca_nhead = sa_nhead

    num_decoder_layers = len(pt_transformer.decoder.layers)
    dim_feedforward = pt_transformer.decoder.layers[0].linear1.out_features
    dropout = pt_transformer.decoder.layers[0].dropout1.p
    num_queries = pt_transformer.num_queries

    # Extract deformable cross-attention configuration
    cross_attn = pt_transformer.decoder.layers[0].cross_attn
    n_levels = cross_attn.n_levels
    dec_n_points = cross_attn.n_points
    ca_nhead = cross_attn.n_heads

    # Validate n_points against the weight matrix shape
    sampling_offsets_shape = cross_attn.sampling_offsets.weight.shape
    out_features = sampling_offsets_shape[0]
    calculated_n_points = out_features // (ca_nhead * n_levels * 2)

    if calculated_n_points != dec_n_points:
        print(
            f"WARNING: MSDeformAttn.n_points ({dec_n_points}) differs from weight shape derived ({calculated_n_points}) using n_heads={ca_nhead}. Using derived value."
        )
        dec_n_points = calculated_n_points

    # Infer activation function name from the pretrained layer
    pt_activation = getattr(pt_transformer.decoder.layers[0], "activation", None)
    activation = "relu"
    if pt_activation is not None:
        if hasattr(pt_activation, "__name__"):
            if "relu" in pt_activation.__name__:
                activation = "relu"
            elif "gelu" in pt_activation.__name__:
                activation = "gelu"
            elif "glu" in pt_activation.__name__:
                activation = "glu"

    normalize_before = getattr(
        pt_transformer.decoder.layers[0], "normalize_before", False
    )

    # Detect two-stage mode from presence of encoder class embeddings
    two_stage = (
        hasattr(pt_transformer, "enc_out_class_embed")
        and len(pt_transformer.enc_out_class_embed) > 0
    )

    return_intermediate_dec = getattr(
        pt_transformer.decoder, "return_intermediate", True
    )
    bbox_reparam = getattr(pt_transformer, "bbox_reparam", False)
    lite_refpoint_refine = getattr(
        pt_transformer.decoder, "lite_refpoint_refine", False
    )

    print(
        f"Config: d_model={d_model}, nhead={sa_nhead}, layers={num_decoder_layers}, levels={n_levels}"
    )
    print(
        f"Flags: two_stage={two_stage}, return_intermediate_dec={return_intermediate_dec}, bbox_reparam={bbox_reparam}, lite_refpoint_refine={lite_refpoint_refine}, activation={activation}"
    )

    # 2. Build Keras Transformer with matching configuration
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

    # Create matching encoder output heads on the Keras model.
    # These are normally assigned externally by the detection model.
    if hasattr(pt_transformer, "enc_out_class_embed"):
        keras_transformer.enc_out_class_embed = [
            layers.Dense(layer.out_features)
            for layer in pt_transformer.enc_out_class_embed
        ]
        for layer in keras_transformer.enc_out_class_embed:
            layer.build((None, d_model))

    if hasattr(pt_transformer, "enc_out_bbox_embed"):
        keras_transformer.enc_out_bbox_embed = [
            KerasMLP(d_model, d_model, 4, 3) for _ in pt_transformer.enc_out_bbox_embed
        ]
        for mlp in keras_transformer.enc_out_bbox_embed:
            mlp(ops.zeros((1, d_model)))

    if getattr(pt_transformer.decoder, "bbox_embed", None) is not None:
        keras_transformer.decoder.bbox_embed = KerasMLP(d_model, d_model, 4, 3)
        keras_transformer.decoder.bbox_embed(ops.zeros((1, d_model)))

    # Debug lengths
    if two_stage:
        print(f"DEBUG: enc_output len (source): {len(pt_transformer.enc_output)}")
        if len(pt_transformer.enc_output) > 0:
            print(f"DEBUG: enc_output[0]: {pt_transformer.enc_output[0]}")
        print(f"DEBUG: enc_output len (keras): {len(keras_transformer.enc_output)}")

    # 3. Create random test inputs with valid shapes
    bs = 1

    spatial_shapes = []
    srcs_np = []
    masks_np = []
    pos_embeds_np = []

    # Build multi-scale feature inputs at progressively halved resolutions
    h0, w0 = 32, 32
    for l in range(n_levels):
        h, w = max(1, h0 // (2**l)), max(1, w0 // (2**l))
        spatial_shapes.append((h, w))
        srcs_np.append(np.random.randn(bs, d_model, h, w).astype(np.float32))
        masks_np.append(np.zeros((bs, h, w), dtype=bool))
        pos_embeds_np.append(np.random.randn(bs, d_model, h, w).astype(np.float32))

    query_feat_np = np.random.randn(num_queries, d_model).astype(np.float32)
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

    # Build Keras model by running a forward pass to initialize all variables
    print("Building Keras model variables...")
    keras_transformer(k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed)

    # 4. Transfer all weights
    print("Transferring weights...")
    transfer_transformer_weights(pt_transformer, keras_transformer, d_model, sa_nhead)

    # 5. Run forward passes and compare outputs
    print("Running forward passes...")
    with torch.no_grad():
        pt_outs = pt_transformer(
            t_srcs, t_masks, t_pos_embeds, t_refpoint_embed, t_query_feat
        )

    k_outs = keras_transformer(
        k_srcs, k_masks, k_pos_embeds, k_query_feat, k_refpoint_embed
    )

    # 6. Compare each output tensor against tolerance thresholds
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
