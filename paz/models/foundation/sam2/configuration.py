"""Immutable Hiera trunk configuration for the four SAM 2 architectures.

SAM 2 and SAM 2.1 share these architectures and differ only in checkpoint
weights, so the eight public factories reuse these four configs. The SAM
prompt encoder, mask decoder, and video memory modules are identical across
every variant and are therefore fixed as module constants instead of
configuration fields.
"""
from collections import namedtuple

IMAGE_SIZE = 1024
BACKBONE_STRIDE = 16
PROMPT_EMBED_DIM = 256
MEMORY_DIM = 64
NUM_MEMORIES = 7
DIM_MUL = 2
HEAD_MUL = 2
QUERY_STRIDE = (2, 2)
QUERY_POOL_STAGES = 3
MLP_RATIO = 4.0

FIELDS = "embed_dim num_heads stages global_blocks window_spec background"
SAM2Config = namedtuple("SAM2Config", FIELDS)

tiny = 96, 1, (1, 2, 7, 2), (5, 7, 9), (8, 4, 14, 7), (7, 7)
small = 96, 1, (1, 2, 11, 2), (7, 10, 13), (8, 4, 14, 7), (7, 7)
base = 112, 2, (2, 3, 16, 3), (12, 16, 20), (8, 4, 14, 7), (14, 14)
large = 144, 2, (2, 6, 36, 4), (23, 33, 43), (8, 4, 16, 8), (7, 7)

TINY = SAM2Config(*tiny)
SMALL = SAM2Config(*small)
BASE_PLUS = SAM2Config(*base)
LARGE = SAM2Config(*large)
