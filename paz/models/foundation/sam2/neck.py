"""FPN neck for SAM 2 (channels-last, functional).

Lateral 1x1 convolutions to ``d_model`` with top-down 2x nearest fusion on
the two lowest-resolution levels, matching the official ``FpnNeck``. The
sinusoidal position encoding it also produces is only consumed by the video
memory path, so it is omitted from image inference.
"""
from keras.layers import Conv2D, UpSampling2D, Add

from paz.models.foundation.sam2.configuration import PROMPT_EMBED_DIM

TOP_DOWN_LEVELS = (2, 3)


def build(features):
    last = len(features) - 1
    laterals = []
    for level, feature in enumerate(features):
        name = f"neck_conv_{last - level}"
        laterals.append(Conv2D(PROMPT_EMBED_DIM, 1, name=name)(feature))
    outputs = [None] * len(features)
    previous = None
    for level in range(last, -1, -1):
        lateral = laterals[level]
        if level in TOP_DOWN_LEVELS and previous is not None:
            upsampled = UpSampling2D(2, interpolation="nearest")(previous)
            previous = Add()([lateral, upsampled])
        else:
            previous = lateral
        outputs[level] = previous
    return outputs
