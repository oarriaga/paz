"""SAM 2 video-tracking application test (opt-in; needs hosted weights)."""
import os

import numpy as np
import pytest

import paz
from paz.models.foundation.sam2.video import Prompt

DOWNLOAD = os.environ.get("PAZ_SAM2_DOWNLOAD")
REASON = "set PAZ_SAM2_DOWNLOAD to fetch weights"


@pytest.mark.skipif(not DOWNLOAD, reason=REASON)
def test_track_sam21_tiny_yields_a_mask_per_frame():
    track = paz.applications.TrackSAMHieraTiny21(draw=False)
    frames = [np.zeros((120, 160, 3), np.uint8) for _ in range(3)]
    prompts = [Prompt(0, 1, points=[[80.0, 60.0]], labels=[1])]
    results = list(track(frames, prompts))
    assert [frame for frame, _ in results] == [0, 1, 2]
    for _, masks in results:
        assert masks.shape == (1, 120, 160)
        assert masks.dtype == bool
