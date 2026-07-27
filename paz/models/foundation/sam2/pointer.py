"""SAM 2 object pointers: what a tracked frame contributes to later frames.

The mask decoder's selected mask token becomes the frame's pointer; frames
where the object is absent fall back to the learned ``no_obj_ptr`` instead. A
pointer also carries how far its frame is from the frame being tracked: the
tracker turns that distance into a sine vector and ``build_time`` projects it
down to the memory dimension.
"""
from keras import Input, Model, ops
from keras.layers import Dense

from paz.models.foundation.sam2.configuration import MEMORY_DIM
from paz.models.foundation.sam2.configuration import PROMPT_EMBED_DIM
from paz.models.foundation.sam2.mask_decoder import mlp


def build(name="sam2_pointer"):
    token = Input((PROMPT_EMBED_DIM,), name="token")
    absent = Input((1,), name="absent")
    args = token, PROMPT_EMBED_DIM, PROMPT_EMBED_DIM, 3, "obj_ptr_proj"
    projected = mlp(*args)
    kwargs = dict(use_bias=False, name="no_obj_ptr")
    missing = Dense(PROMPT_EMBED_DIM, **kwargs)(absent)
    pointer = ops.add(projected * (1.0 - absent), missing)
    return Model((token, absent), pointer, name=name)


def build_time(name="sam2_pointer_time"):
    distance = Input((PROMPT_EMBED_DIM,), name="distance")
    projected = Dense(MEMORY_DIM, name="obj_ptr_tpos_proj")(distance)
    return Model(distance, projected, name=name)
