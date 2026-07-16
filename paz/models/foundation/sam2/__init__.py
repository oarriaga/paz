"""SAM 2 / SAM 2.1 static-image models built from PAZ transformer parts.

One shared architecture (Hiera trunk, FPN neck, prompt encoder, two-way mask
decoder) drives the eight official factories, which differ only by immutable
configuration and checkpoint weights. Video memory is intentionally out of
scope for this module.
"""
