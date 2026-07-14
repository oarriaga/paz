import keras

from paz.models.transformers.logits import soft_cap as apply_soft_cap
from paz.models.foundation.gemma4.configuration import TextBackboneArgs
from paz.models.foundation.gemma4.model import Gemma4Backbone


@keras.saving.register_keras_serializable(package="gemma4")
class Gemma4CausalLM(keras.Model):
    """Wraps Gemma4Backbone with reverse-embedding logits and final soft-cap.

    `call` scores a full sequence; `call_with_cache` runs one cached step for
    generation. Both share the backbone's single weight set.
    """

    def __init__(self, config, name="gemma4_causal_lm", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.backbone = Gemma4Backbone(config)

    def call(self, inputs):
        return self.logits(self.backbone(inputs))

    def call_with_cache(self, input_embedding, cache, index, positions=None,
                        per_layer_full=None):
        hidden, cache = self.backbone.call_with_cache(
            input_embedding, cache, index, positions, per_layer_full)
        return self.logits(hidden), cache

    def logits(self, hidden):
        logits = self.backbone.token_embedding(hidden, reverse=True)
        return apply_soft_cap(logits, self.config.final_logit_soft_cap)

    def build_cache(self, max_length, batch_size=1):
        return self.backbone.build_cache(max_length, batch_size)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config._asdict()
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["config"] = TextBackboneArgs(**config["config"])
        return cls(**config)
