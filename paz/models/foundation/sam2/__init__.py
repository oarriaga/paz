"""SAM 2 / SAM 2.1 static-image models built from PAZ transformer parts.

One shared architecture (Hiera trunk, FPN neck, prompt encoder, two-way mask
decoder) drives the eight official factories, which differ only by immutable
configuration and checkpoint weights. Video memory is intentionally out of
scope for this module.
"""
from paz.models.foundation.sam2.pretrained import SAM2HieraTiny
from paz.models.foundation.sam2.pretrained import SAM2HieraSmall
from paz.models.foundation.sam2.pretrained import SAM2HieraBasePlus
from paz.models.foundation.sam2.pretrained import SAM2HieraLarge
from paz.models.foundation.sam2.pretrained import SAM21HieraTiny
from paz.models.foundation.sam2.pretrained import SAM21HieraSmall
from paz.models.foundation.sam2.pretrained import SAM21HieraBasePlus
from paz.models.foundation.sam2.pretrained import SAM21HieraLarge
