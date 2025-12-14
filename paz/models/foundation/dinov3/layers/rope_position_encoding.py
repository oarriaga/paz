import keras
from keras import ops, random
import math
import numpy as np
from keras.initializers import Initializer


@keras.saving.register_keras_serializable(package="MyInitializers")
class RopePeriodsInitializer(Initializer):
    def __init__(self, D_head, base, min_period, max_period):
        self.D_head = D_head
        self.base = base
        self.min_period = min_period
        self.max_period = max_period

    def __call__(self, shape, dtype=None):
        dim_div_4 = self.D_head // 4
        if self.base is not None:
            arange_tensor = ops.arange(dim_div_4, dtype=dtype)
            exponents = 2 * arange_tensor / (self.D_head // 2)
            new_periods = ops.power(self.base, exponents)
        else:
            base = self.max_period / self.min_period
            exponents = ops.linspace(0, 1, dim_div_4, dtype=dtype)
            new_periods = ops.power(base, exponents)
            new_periods = new_periods / base
            new_periods = new_periods * self.max_period
        return new_periods

    def get_config(self):
        return {
            "D_head": self.D_head,
            "base": self.base,
            "min_period": self.min_period,
            "max_period": self.max_period,
        }


@keras.saving.register_keras_serializable(package="paz.dinov3")
class RopePositionEmbedding(keras.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: str = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not (embed_dim % (4 * num_heads) == 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by 4 * num_heads ({4*num_heads})."
            )

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.D_head = embed_dim // num_heads

    def build(self, input_shape=None):
        self.periods = self.add_weight(
            shape=(self.D_head // 4,),
            initializer=RopePeriodsInitializer(
                self.D_head, self.base, self.min_period, self.max_period
            ),
            trainable=False,
            name="periods",
            dtype=self.dtype,
        )
        self._initialize_periods()
        self.built = True

    def _initialize_periods(self):
        dim_div_4 = self.D_head // 4
        if self.base is not None:
            arange_tensor = ops.arange(dim_div_4, dtype=self.dtype)
            exponents = 2 * arange_tensor / (self.D_head // 2)
            new_periods = ops.power(self.base, exponents)
        else:
            base = self.max_period / self.min_period
            exponents = ops.linspace(0, 1, dim_div_4, dtype=self.dtype)
            new_periods = ops.power(base, exponents)
            new_periods = new_periods / base
            new_periods = new_periods * self.max_period
        self.periods.assign(new_periods)

    def call(self, H: int, W: int, training: bool = False):
        coords_h_int = ops.arange(H, dtype="int32")
        coords_w_int = ops.arange(W, dtype="int32")
        coords_h = ops.cast(coords_h_int, dtype=self.dtype) + 0.5
        coords_w = ops.cast(coords_w_int, dtype=self.dtype) + 0.5
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = coords_h / max_HW
            coords_w = coords_w / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = coords_h / min_HW
            coords_w = coords_w / min_HW
        elif self.normalize_coords == "separate":
            coords_h = coords_h / H
            coords_w = coords_w / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        grid_h, grid_w = ops.meshgrid(coords_h, coords_w, indexing="ij")
        coords = ops.stack([grid_h, grid_w], axis=-1)
        coords = ops.reshape(coords, (-1, 2))
        coords = 2.0 * coords - 1.0
        if training and self.shift_coords is not None:
            shift_hw = random.uniform(
                (2,),
                minval=-self.shift_coords,
                maxval=self.shift_coords,
                dtype=self.dtype,
            )
            coords += shift_hw
        if training and self.jitter_coords is not None:
            jitter_max = math.log(self.jitter_coords)
            jitter_min = -jitter_max
            log_jitter = random.uniform(
                (2,), minval=jitter_min, maxval=jitter_max, dtype=self.dtype
            )
            coords *= ops.exp(log_jitter)
        if training and self.rescale_coords is not None:
            rescale_max = math.log(self.rescale_coords)
            rescale_min = -rescale_max
            log_rescale = random.uniform(
                (1,), minval=rescale_min, maxval=rescale_max, dtype=self.dtype
            )
            coords *= ops.exp(log_rescale)

        coords = ops.expand_dims(coords, axis=-1)
        periods_exp = ops.expand_dims(ops.expand_dims(self.periods, axis=0), axis=0)

        angles = 2.0 * math.pi * coords / periods_exp

        angles = ops.reshape(angles, (-1, self.D_head // 2))
        angles = ops.tile(angles, [1, 2])

        cos = ops.cos(angles)
        sin = ops.sin(angles)
        return (sin, cos)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "base": self.base,
                "min_period": self.min_period,
                "max_period": self.max_period,
                "normalize_coords": self.normalize_coords,
                "shift_coords": self.shift_coords,
                "jitter_coords": self.jitter_coords,
                "rescale_coords": self.rescale_coords,
            }
        )
        return config
