import os
import pytest
import numpy as np
import keras
import tempfile
from paz.models.foundation.dinov3.models.vision_transformer import (
    vit_small,
    DINOV3VITS,
    DINOV3VITB,
    DINOV3VITL,
    DinoVisionTransformer,
)


def test_serialization():
    print("\n--- Testing Model Serialization ---")
    # 1. Instantiate a small model
    model = vit_small(
        patch_size=16,
        embed_dim=192,  # Smaller for speed
        depth=2,
        num_heads=3,
        input_shape=(224, 224, 3),
    )

    # 2. Build and run inference
    input_data = np.random.randn(1, 224, 224, 3).astype("float32")
    output_original = model(input_data, training=False)

    # 3. Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        save_path = tmp.name

    try:
        model.save(save_path)
        print(f"Model saved to {save_path}")

        # 4. Load back
        # Clear custom objects to ensure registration worked
        # (Though in same process, registration is global)
        loaded_model = keras.models.load_model(save_path)
        print("Model loaded successfully")

        # 5. Compare config
        # (Basic check)
        assert isinstance(loaded_model, DinoVisionTransformer)
        assert loaded_model.embed_dim == model.embed_dim

        # 6. Compare output
        output_loaded = loaded_model(input_data, training=False)
        np.testing.assert_allclose(
            output_original, output_loaded, rtol=1e-5, atol=1e-5
        )
        print("✅ Serialization test passed: Outputs match.")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


def test_pretrained_instantiation():
    print("\n--- Testing Pretrained Model Instantiation ---")

    # We just check if they can be instantiated without crashing.
    # If weights are missing, it prints a warning but returns the model.

    try:
        model_s = DINOV3VITS()
        assert isinstance(model_s, DinoVisionTransformer)
        print("✅ DINOV3VITS instantiated.")

        model_b = DINOV3VITB()
        assert isinstance(model_b, DinoVisionTransformer)
        print("✅ DINOV3VITB instantiated.")

        model_l = DINOV3VITL()
        assert isinstance(model_l, DinoVisionTransformer)
        print("✅ DINOV3VITL instantiated.")

    except Exception as e:
        pytest.fail(f"Instantiation failed: {e}")


if __name__ == "__main__":
    test_serialization()
    test_pretrained_instantiation()
