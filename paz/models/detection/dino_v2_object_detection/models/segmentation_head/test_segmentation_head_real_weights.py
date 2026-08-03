import os
import sys
import numpy as np
import pytest
import torch

# Ensure project root is on the import path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))  # fmt: skip
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from rfdetr.detr import (
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegXLarge,
        RFDETRSeg2XLarge,
        RFDETRSegPreview,
    )
except ImportError:
    rfdetr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../examples/rf-detr_original_pytorch_implementation'))  # fmt: skip
    if rfdetr_path not in sys.path:
        sys.path.insert(0, rfdetr_path)
    from rfdetr.detr import (
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegXLarge,
        RFDETRSeg2XLarge,
        RFDETRSegPreview,
    )

from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import (  # fmt: skip
    SegmentationHead,
    apply_segmentation_head,
)
from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_weights_porting_utils import (  # fmt: skip
    copy_segmentation_head,
    assert_allclose,
)

MODEL_VARIANTS = {
    "Nano": RFDETRSegNano,
    "Small": RFDETRSegSmall,
    "Preview": RFDETRSegPreview,
}
MODEL_VARIANTS["XLarge"] = RFDETRSegXLarge
MODEL_VARIANTS["2XLarge"] = RFDETRSeg2XLarge


def extract_pt_segmentation_head(model_class):
    print(f"Loading pretrained {model_class.__name__}...")
    model = model_class(pretrained=True)
    pt_model = model.model.model
    pt_head = pt_model.segmentation_head
    pt_head.eval()
    pt_head.cpu()
    return pt_head, model.model_config


@pytest.mark.parametrize("variant_name", list(MODEL_VARIANTS.keys()))
def test_segmentation_head_real_weights(variant_name):
    print(f"\n{'='*60}")
    print(f"Testing SegmentationHead parity for RFDETR {variant_name}")
    print(f"{'='*60}")

    model_cls = MODEL_VARIANTS[variant_name]

    # 1. Load reference head and run it in float64 for higher precision
    pt_head, config = extract_pt_segmentation_head(model_cls)
    pt_head = pt_head.double()

    # 2. Build Keras head with matching configuration
    hidden_dim = config.hidden_dim
    dec_layers = config.dec_layers
    mask_downsample_ratio = config.mask_downsample_ratio
    bottleneck_ratio = 1

    print(f"Configuration: hidden_dim={hidden_dim}, dec_layers={dec_layers}, downsample={mask_downsample_ratio}")  # fmt: skip

    keras_head = SegmentationHead(
        hidden_dim,
        dec_layers,
        bottleneck_ratio=bottleneck_ratio,
        downsample_ratio=mask_downsample_ratio,
    )

    image_size = (config.resolution, config.resolution)
    spatial_shape = (1, hidden_dim, image_size[0] // 32, image_size[1] // 32)

    # 3. Transfer weights from reference to Keras
    print("Copying weights...")
    copy_segmentation_head(pt_head, keras_head)

    # 4. Run both implementations on identical random inputs
    spatial_np = np.random.randn(*spatial_shape).astype(np.float32)
    qf_np = [np.random.randn(1, 10, hidden_dim).astype(np.float32) for _ in range(dec_layers)]  # fmt: skip

    print("\n--- Verifying Full Head ---")

    with torch.no_grad():
        pt_qfs = [torch.from_numpy(q).double() for q in qf_np]
        spatial_tensor = torch.from_numpy(spatial_np).double()
        pt_out = pt_head(spatial_tensor, pt_qfs, image_size)
    keras_out = apply_segmentation_head(keras_head, spatial_np, qf_np, image_size=image_size)  # fmt: skip

    assert_allclose(pt_out, keras_out, atol=5e-4, rtol=1e-4)
    print(f"RFDETR {variant_name} SegmentationHead Verification PASSED!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        variant = sys.argv[1]
        if variant in MODEL_VARIANTS:
            test_segmentation_head_real_weights(variant)
        else:
            available = list(MODEL_VARIANTS.keys())
            print(f"Unknown variant {variant}. Available: {available}")
    else:
        pytest.main([__file__, "-v", "-s"])
