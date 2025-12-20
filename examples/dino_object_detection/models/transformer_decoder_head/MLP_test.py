import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import keras
from keras import layers
import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(f"Project Root: {project_root}")
print(f"Script Dir: {script_dir}")
sys.path.append(project_root)
# -------------------------------------------------------------------------
# PyTorch Implementation
# -------------------------------------------------------------------------
from examples.dino_object_detection.models.transformer_decoder_head.torch_transformer_for_testing import (
    MLP as MLP_PyTorch,
)

# -------------------------------------------------------------------------
# Keras 3 Implementation
# -------------------------------------------------------------------------
from examples.dino_object_detection.models.transformer_decoder_head.MLP import (
    MLP,
)


# -------------------------------------------------------------------------
# 3. The PyTest Case
# -------------------------------------------------------------------------
def test_mlp_equivalence():
    # --- Configuration ---
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    num_layers = 3
    batch_size = 8

    # --- Instantiate Models ---
    pt_model = MLP_PyTorch(input_dim, hidden_dim, output_dim, num_layers)
    k_model = MLP(input_dim, hidden_dim, output_dim, num_layers)

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


if __name__ == "__main__":
    # test_mlp_equivalence()
    pytest.main(["-v", __file__])
