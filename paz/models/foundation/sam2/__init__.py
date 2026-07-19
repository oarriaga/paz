"""SAM 2 / SAM 2.1 static-image models built from PAZ transformer parts.

One shared architecture (Hiera trunk, FPN neck, prompt encoder, two-way mask
decoder) drives the eight official factories, which differ only by immutable
configuration and checkpoint weights. Video memory is intentionally out of
scope for this module.
"""
from paz.models.foundation.sam2.pretrained import SAMHieraTiny2
from paz.models.foundation.sam2.pretrained import SAMHieraSmall2
from paz.models.foundation.sam2.pretrained import SAMHieraBasePlus2
from paz.models.foundation.sam2.pretrained import SAMHieraLarge2
from paz.models.foundation.sam2.pretrained import SAMHieraTiny21
from paz.models.foundation.sam2.pretrained import SAMHieraSmall21
from paz.models.foundation.sam2.pretrained import SAMHieraBasePlus21
from paz.models.foundation.sam2.pretrained import SAMHieraLarge21
