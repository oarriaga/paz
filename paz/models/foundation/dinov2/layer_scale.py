import keras
import keras.ops as ops
from typing import Union


class LayerScale(keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
        dtype: Union[str, keras.mixed_precision.Policy, None] = None,
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.init_values = init_values
        self.inplace = inplace

        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        return ops.multiply(x, self.gamma)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "init_values": self.init_values,
                "inplace": self.inplace,
            }
        )
        return config
