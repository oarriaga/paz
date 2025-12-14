import numpy as np
import torch
import os
import pytest
import keras
from keras import ops
import convnext
import torch_convnext_for_testing

TOLERANCE = 1e-5
DINO_WEIGHTS_DIR = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\paz/"

MODEL_PARAMS = [
    (
        "tiny",
        torch_convnext_for_testing.PT_get_convnext_arch("convnext_tiny"),
        convnext.get_convnext_arch("convnext_tiny"),
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    ),
    (
        "small",
        torch_convnext_for_testing.PT_get_convnext_arch("convnext_small"),
        convnext.get_convnext_arch("convnext_small"),
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    ),
    (
        "base",
        torch_convnext_for_testing.PT_get_convnext_arch("convnext_base"),
        convnext.get_convnext_arch("convnext_base"),
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    ),
    (
        "large",
        torch_convnext_for_testing.PT_get_convnext_arch("convnext_large"),
        convnext.get_convnext_arch("convnext_large"),
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    ),
]


def port_weights(pt_model, keras_model):
    """
    Ports weights from a PyTorch model to an equivalent Keras 3 model.
    This function relies on the models having their parameters
    in the exact same order.
    """
    print("Starting weight porting...")

    pt_params = list(pt_model.parameters())
    keras_weights = keras_model.weights
    trainable_keras_weights = [w for w in keras_weights if w.trainable]

    if len(pt_params) != len(trainable_keras_weights):
        print(f"PyTorch param count: {len(pt_params)}")
        print(f"Keras trainable weight count: {len(trainable_keras_weights)}")
        raise ValueError("Model parameter counts do not match! Cannot port weights.")

    pt_param_idx = 0
    for kw in keras_weights:
        if not kw.trainable:
            continue

        pt_p = pt_params[pt_param_idx]
        pt_p_np = pt_p.detach().cpu().numpy()

        if "kernel" in kw.name:
            if len(kw.shape) == 4:
                if kw.shape[3] == 1:
                    pt_p_np = pt_p_np.transpose(2, 3, 0, 1)
                else:
                    pt_p_np = pt_p_np.transpose(2, 3, 1, 0)
            elif len(kw.shape) == 2:
                pt_p_np = pt_p_np.transpose(1, 0)

        if kw.shape != pt_p_np.shape:
            raise ValueError(
                f"Shape mismatch for weight '{kw.name}':\n"
                f"Keras shape: {kw.shape}\n"
                f"PyTorch shape (after transpose): {pt_p_np.shape}"
            )

        kw.assign(pt_p_np)
        pt_param_idx += 1

    if pt_param_idx != len(pt_params):
        raise Exception("Did not port all PyTorch parameters!")
    print("Weight porting complete.")


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\nSeeds set to {seed}")


def assert_all_close(test_name, pt_out, keras_out, tolerance=TOLERANCE):
    """Helper function to compare two tensors."""
    if isinstance(pt_out, torch.Tensor):
        pt_out_np = pt_out.detach().cpu().numpy()
    else:
        pt_out_np = pt_out

    if isinstance(keras_out, keras.KerasTensor):
        keras_out_np = keras.ops.convert_to_numpy(keras_out)
    else:
        keras_out_np = keras_out

    try:
        mean_abs_diff = np.mean(np.abs(pt_out_np - keras_out_np))
        if mean_abs_diff > tolerance:
            raise AssertionError(
                f"Mean absolute difference {mean_abs_diff} exceeds tolerance {tolerance}"
            )
        # npt.assert_allclose(pt_out_np, keras_out_np, rtol=1e-4)

        print(f"✅ [PASS] {test_name}")
    except AssertionError as e:
        print(f"❌ [FAIL] {test_name}")
        print(f"Max abs diff: {np.max(np.abs(pt_out_np - keras_out_np))}")
        print(f"Mean abs diff: {np.mean(np.abs(pt_out_np - keras_out_np))}")
        print(e)
        raise e


def transfer_weights_convnext(pt_model, keras_model):
    print("Starting by-name weight transfer...")
    pt_sd = pt_model.state_dict()

    try:
        keras_model.downsample_layers[0].layers[0].set_weights(
            [
                pt_sd["downsample_layers.0.0.weight"].permute(2, 3, 1, 0).numpy(),
                pt_sd["downsample_layers.0.0.bias"].numpy(),
            ]
        )
        keras_model.downsample_layers[0].layers[1].set_weights(
            [
                pt_sd["downsample_layers.0.1.weight"].numpy(),
                pt_sd["downsample_layers.0.1.bias"].numpy(),
            ]
        )

        # 3 Downsampling layers (downsample_layers[1, 2, 3])
        for i in range(1, 4):
            keras_model.downsample_layers[i].layers[0].set_weights(
                [
                    pt_sd[f"downsample_layers.{i}.0.weight"].numpy(),
                    pt_sd[f"downsample_layers.{i}.0.bias"].numpy(),
                ]
            )
            keras_model.downsample_layers[i].layers[1].set_weights(
                [
                    pt_sd[f"downsample_layers.{i}.1.weight"]
                    .permute(2, 3, 1, 0)
                    .numpy(),
                    pt_sd[f"downsample_layers.{i}.1.bias"].numpy(),
                ]
            )

        # 4 Stages
        for i in range(4):
            for j in range(len(keras_model.stages[i].layers)):
                block = keras_model.stages[i].layers[j]
                pt_prefix = f"stages.{i}.{j}"
                block.dwconv.set_weights(
                    [
                        pt_sd[f"{pt_prefix}.dwconv.weight"].permute(2, 3, 0, 1).numpy(),
                        pt_sd[f"{pt_prefix}.dwconv.bias"].numpy(),
                    ]
                )
                block.norm.set_weights(
                    [
                        pt_sd[f"{pt_prefix}.norm.weight"].numpy(),
                        pt_sd[f"{pt_prefix}.norm.bias"].numpy(),
                    ]
                )
                block.pwconv1.set_weights(
                    [
                        pt_sd[f"{pt_prefix}.pwconv1.weight"].T.numpy(),
                        pt_sd[f"{pt_prefix}.pwconv1.bias"].numpy(),
                    ]
                )
                block.pwconv2.set_weights(
                    [
                        pt_sd[f"{pt_prefix}.pwconv2.weight"].T.numpy(),
                        pt_sd[f"{pt_prefix}.pwconv2.bias"].numpy(),
                    ]
                )
                if block.gamma is not None:
                    block.gamma.assign(pt_sd[f"{pt_prefix}.gamma"].numpy())

        keras_model.norm.set_weights(
            [pt_sd["norm.weight"].numpy(), pt_sd["norm.bias"].numpy()]
        )

        print("By-name weight transfer complete.")

    except KeyError as e:
        print(f"❌ [FAIL] Weight transfer failed. Missing key: {e}")
        print(
            "This often means the Keras and PyTorch model structures are out of sync."
        )
        raise e
    except Exception as e:
        print(f"❌ [FAIL] Weight transfer failed with an unexpected error: {e}")
        raise e


@pytest.fixture(scope="module", params=MODEL_PARAMS, ids=[p[0] for p in MODEL_PARAMS])
def setup_models(request):
    """
    Module-scoped fixture to set up models, inputs, and port weights.
    This runs ONCE for EACH parameter set (tiny, small, base, large).
    """
    set_seeds(42)
    arch_name, pt_arch_fn, keras_arch_fn, weight_path = request.param

    print(f"\n--- Setting up for model: {arch_name} ---")

    # --- 1. Check for weight file ---
    if not os.path.isfile(weight_path):
        pytest.skip(f"Weight file not found, skipping: {weight_path}")

    # --- 2. Instantiate PyTorch Model ---
    print("Instantiating PyTorch model...")
    pt_model = pt_arch_fn(patch_size=16)

    # Load the pre-trained weights
    print(f"Loading PyTorch weights from: {weight_path}")
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=True)
    pt_model.eval()
    print("PyTorch model instantiated and weights loaded.")

    # --- 3. Instantiate Keras Model ---
    print("Instantiating Keras model...")
    keras_model = keras_arch_fn(patch_size=16)

    # --- 4. Create Test Input ---
    # Models are channels-first: (N, C, H, W)
    N, C, H, W = 2, 3, 224, 224
    print(f"Creating random input tensor: ({N}, {C}, {H}, {W})")
    np_input = np.random.rand(N, C, H, W).astype("float32")
    torch_input = torch.from_numpy(np_input)
    keras_input = keras.ops.convert_to_tensor(np_input)

    # --- 5. Build Keras Model & Port Weights ---
    try:
        _ = keras_model(keras_input, training=False)
        print("Keras model built.")
    except Exception as e:
        print("\n--- Keras Model Build FAILED ---")
        pytest.fail(f"Keras model build failed for {arch_name}: {e}")

    try:
        transfer_weights_convnext(pt_model, keras_model)
    except Exception as e:
        print("\n--- Weight Porting FAILED ---")
        pytest.fail(f"Weight porting failed for {arch_name}: {e}")

    print(f"--- Model Setup Complete for: {arch_name} ---")

    # Yield all necessary components to the tests
    return {
        "arch_name": arch_name,
        "pt_model": pt_model,
        "keras_model": keras_model,
        "torch_input": torch_input,
        "keras_input": keras_input,
    }


def test_final_output_equivalence(setup_models):
    """
    Test 1: Compare the final model output (inference mode)
    """
    arch_name = setup_models["arch_name"]
    print(f"\nRunning Test 1 (Final Output) for: {arch_name}")

    pt_out_call = setup_models["pt_model"](
        setup_models["torch_input"], is_training=False
    )
    keras_out_call = setup_models["keras_model"](
        setup_models["keras_input"], training=False
    )

    assert_all_close(
        f"Final 'call' (inference) - {arch_name}",
        pt_out_call,
        keras_out_call,
        tolerance=1e-4,
    )


def test_deep_dive_block_equivalence(setup_models):
    """
    Test 2: Compare the output of each layer in each block
    """
    arch_name = setup_models["arch_name"]
    keras_model = setup_models["keras_model"]
    pt_model = setup_models["pt_model"]

    print(f"\nRunning Test 2 (Deep Dive) for: {arch_name}")

    any_failure = False

    def _compare_and_correct(keras_tensor, torch_tensor, layer_name):
        nonlocal any_failure
        torch_np = torch_tensor.detach().cpu().numpy()
        keras_np = keras.ops.convert_to_numpy(keras_tensor)
        try:
            assert_all_close(layer_name, torch_np, keras_np, tolerance=1e-4)
            return keras_tensor
        except AssertionError as e:
            any_failure = True
            print(f"       Correcting Keras tensor to continue analysis...")
            return keras.ops.convert_to_tensor(torch_np)

    np.random.seed(42)
    np_input = np.random.rand(2, 3, 224, 224).astype("float32")
    x_keras = keras.ops.convert_to_tensor(np_input)
    x_torch = torch.from_numpy(np_input)

    for i in range(4):
        print(f"\n--- Analyzing Stage {i} ({arch_name}) ---")

        # --- 1. Downsample Layer ---
        keras_down_layer = keras_model.downsample_layers[i]
        pt_down_layer = pt_model.downsample_layers[i]

        x_keras = keras_down_layer(x_keras)
        x_torch = pt_down_layer(x_torch)

        x_keras = _compare_and_correct(
            x_keras, x_torch, f"Stage {i} - Downsample Output"
        )

        # --- 2. Blocks in Stage ---
        for j in range(len(keras_model.stages[i].layers)):
            keras_block = keras_model.stages[i].layers[j]
            pt_block = pt_model.stages[i][j]

            input_keras = x_keras
            input_torch = x_torch

            layer_prefix = f"Stage {i} Block {j}"

            # 1. DwConv
            k_dwconv = keras_block.dwconv(input_keras)
            pt_dwconv = pt_block.dwconv(input_torch)
            k_dwconv = _compare_and_correct(
                k_dwconv, pt_dwconv, f"{layer_prefix} - dwconv"
            )

            # 2. Permute
            k_perm1 = ops.transpose(k_dwconv, (0, 2, 3, 1))
            pt_perm1 = pt_dwconv.permute(0, 2, 3, 1)

            # 3. Norm
            k_norm = keras_block.norm(k_perm1)
            pt_norm = pt_block.norm(pt_perm1)
            k_norm = _compare_and_correct(k_norm, pt_norm, f"{layer_prefix} - norm")

            # 4. PwConv1
            k_pw1 = keras_block.pwconv1(k_norm)
            pt_pw1 = pt_block.pwconv1(pt_norm)
            k_pw1 = _compare_and_correct(k_pw1, pt_pw1, f"{layer_prefix} - pwconv1")

            # 5. Act
            k_act = keras_block.act(k_pw1)
            pt_act = pt_block.act(pt_pw1)
            k_act = _compare_and_correct(k_act, pt_act, f"{layer_prefix} - act")

            # 6. PwConv2
            k_pw2 = keras_block.pwconv2(k_act)
            pt_pw2 = pt_block.pwconv2(pt_act)
            k_pw2 = _compare_and_correct(k_pw2, pt_pw2, f"{layer_prefix} - pwconv2")

            # 7. Gamma (LayerScale)
            if keras_block.gamma is not None:
                k_pw2 = keras_block.gamma * k_pw2
                pt_pw2 = pt_block.gamma * pt_pw2
                k_pw2 = _compare_and_correct(k_pw2, pt_pw2, f"{layer_prefix} - gamma")

            # 8. Permute Back
            k_perm2 = ops.transpose(k_pw2, (0, 3, 1, 2))
            pt_perm2 = pt_pw2.permute(0, 3, 1, 2)

            # 9. DropPath (is Identity in eval mode)
            k_drop = keras_block.drop_path(k_perm2, training=False)
            pt_drop = pt_block.drop_path(pt_perm2)  # training=False by pt_model.eval()
            k_drop = _compare_and_correct(
                k_drop, pt_drop, f"{layer_prefix} - drop_path"
            )

            # 10. Residual
            x_keras = input_keras + k_drop
            x_torch = input_torch + pt_drop
            x_keras = _compare_and_correct(
                x_keras, x_torch, f"{layer_prefix} - residual_add"
            )

    print(f"\n--- Deep-Dive Analysis Complete for: {arch_name} ---")
    assert (
        not any_failure
    ), f"One or more internal layers for {arch_name} had an output mismatch. See logs for details."
