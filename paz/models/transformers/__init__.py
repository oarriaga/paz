"""Reusable transformer / sequence-model primitives.

Surfaced as ``paz.transformers``. Each submodule names the object it acts on
and exposes bare-verb functions: ``cache.build``, ``cache.update``,
``mask.causal``, ``logits.apply_temperature``, ``search.build``,
``search.greedy``, ``feedforward.gelu``.
"""
from paz.models.transformers import cache
from paz.models.transformers import mask
from paz.models.transformers import logits
from paz.models.transformers import search
from paz.models.transformers import feedforward
