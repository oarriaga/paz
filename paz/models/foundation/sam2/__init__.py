"""SAM 2 / SAM 2.1 image and video models built from PAZ transformer parts.

One shared architecture (Hiera trunk, FPN neck, prompt encoder, two-way mask
decoder) drives the eight image factories, which differ only by immutable
configuration and checkpoint weights. The eight ``*Video`` factories add the
memory encoder, memory attention, and object pointer projections that
``video.track`` needs to follow objects through a video.
"""
from paz.models.foundation.sam2.pretrained import SAMHieraTiny2
from paz.models.foundation.sam2.pretrained import SAMHieraSmall2
from paz.models.foundation.sam2.pretrained import SAMHieraBasePlus2
from paz.models.foundation.sam2.pretrained import SAMHieraLarge2
from paz.models.foundation.sam2.pretrained import SAMHieraTiny21
from paz.models.foundation.sam2.pretrained import SAMHieraSmall21
from paz.models.foundation.sam2.pretrained import SAMHieraBasePlus21
from paz.models.foundation.sam2.pretrained import SAMHieraLarge21
from paz.models.foundation.sam2.pretrained import SAMHieraTiny2Video
from paz.models.foundation.sam2.pretrained import SAMHieraSmall2Video
from paz.models.foundation.sam2.pretrained import SAMHieraBasePlus2Video
from paz.models.foundation.sam2.pretrained import SAMHieraLarge2Video
from paz.models.foundation.sam2.pretrained import SAMHieraTiny21Video
from paz.models.foundation.sam2.pretrained import SAMHieraSmall21Video
from paz.models.foundation.sam2.pretrained import SAMHieraBasePlus21Video
from paz.models.foundation.sam2.pretrained import SAMHieraLarge21Video
