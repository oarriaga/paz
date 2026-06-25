from keras import ops

from paz.models.transformers import mask


def build_attention_mask(padding_mask, bidirectional, sliding_window_size):
    if padding_mask is None:
        return None
    if bidirectional:
        return build_bidirectional_mask(padding_mask)
    positions = build_positions(padding_mask)
    causal_mask = mask.causal(positions, positions)
    if sliding_window_size is not None:
        window = mask.sliding_window(positions, positions, sliding_window_size)
        causal_mask = ops.logical_and(causal_mask, window)
    decoder_mask = merge_padding_mask(padding_mask)
    return ops.logical_and(causal_mask, decoder_mask)


def build_positions(padding_mask):
    ones = ops.ones_like(padding_mask, dtype="int32")
    return ops.cumsum(ones, axis=1) - 1


def build_bidirectional_mask(padding_mask):
    if padding_mask is None:
        return None
    mask = merge_padding_mask(padding_mask)
    return ops.logical_and(mask, ops.transpose(mask, (0, 2, 1)))


def merge_padding_mask(padding_mask):
    if padding_mask is None:
        return None
    mask = ops.cast(padding_mask, "bool")
    return ops.expand_dims(mask, axis=1)
