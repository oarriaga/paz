"""Depth Anything 3 built from canonical DINOv2 and transformer primitives.

The any-view constructors return a Keras model whose outputs are, in order:
``depth, depth_confidence, extrinsics, intrinsics, rays, ray_confidence``.
The monocular constructors return ``depth, sky``.
"""
from .models import DepthAnything3Small
from .models import DepthAnything3Base
from .models import DepthAnything3MonoLarge
from .models import DepthAnything3MetricLarge
from .models import build_da3_small_backbone
