from pathlib import Path

from keras import Model

from .layers.core import apply_tanh_soft_cap
from .model import Gemma4TextBackbone


def Gemma4CausalLM(config, weights_path=None, name="gemma4_causal_lm"):
    backbone = Gemma4TextBackbone(config)
    inputs = backbone.input
    hidden = backbone(inputs)
    embedding = backbone.get_layer("token_embedding")
    logits = embedding(hidden, reverse=True)
    logits = apply_tanh_soft_cap(logits, config.final_logit_soft_cap)
    model = Model(inputs, logits, name=name)
    if weights_path is not None:
        model.load_weights(str(Path(weights_path)))
    return model
