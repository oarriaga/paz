"""SAM 2 segmentation application test (opt-in; needs hosted weights)."""
import os

import numpy as np
import pytest

import paz

DOWNLOAD = os.environ.get("PAZ_SAM2_DOWNLOAD")
REASON = "set PAZ_SAM2_DOWNLOAD to fetch weights"


@pytest.mark.skipif(not DOWNLOAD, reason=REASON)
def test_segment_sam21_tiny_returns_masks():
    segment = paz.applications.SAMHieraTiny21(draw=False)
    image = np.zeros((120, 160, 3), np.uint8)
    masks, scores = segment(image, points=[[80.0, 60.0]], labels=[1])
    assert masks.shape == (3, 120, 160)
    assert masks.dtype == bool
    assert scores.shape == (3,)
