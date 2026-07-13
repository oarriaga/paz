"""Depth Anything 3 built from canonical DINOv2 and transformer primitives.

``build_da3_small`` returns a Keras model whose outputs are, in order:
``depth, depth_confidence, extrinsics, intrinsics, rays, ray_confidence``.
"""
from paz.models.foundation.depth_anything3.models import build_da3_small
from paz.models.foundation.depth_anything3.models import build_da3_small_backbone
