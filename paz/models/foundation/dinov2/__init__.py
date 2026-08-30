"""Canonical function-only DINOv2 built from reusable transformer primitives.

Standard models return ``(class_token, patch_tokens)``. Feature models return
a plain tuple of channels-last feature maps in the requested layer order.
"""
from paz.models.foundation.dinov2.models import DINOv2Small
from paz.models.foundation.dinov2.models import DINOv2Base
from paz.models.foundation.dinov2.models import DINOv2Large
from paz.models.foundation.dinov2.models import DINOv2SmallFeatures
from paz.models.foundation.dinov2.models import DINOv2SmallWindowedFeatures
from paz.models.foundation.dinov2.models import DINOv2BaseFeatures
from paz.models.foundation.dinov2.models import DINOv2LargeFeatures
