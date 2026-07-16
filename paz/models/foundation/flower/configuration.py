from collections import namedtuple

FLOWER_FIELDS = (
    "context_dim hidden_dim num_layers num_heads head_dim mlp_dim "
    "adaln_dim num_shared_signals action_dim num_actions action_space "
    "rope_wavelength rope_max_positions sinusoidal_dim "
    "flow_time_max_period flow_time_frequency_scale frequency_max_period "
    "control_frequency num_sampling_steps"
)
FlowerArgs = namedtuple("FlowerArgs", FLOWER_FIELDS)

CONFIGS = {
    "flower_libero_object": {
        "context_dim": 1024,
        "hidden_dim": 1024,
        "num_layers": 18,
        "num_heads": 16,
        "head_dim": 64,
        "mlp_dim": 2816,
        "adaln_dim": 256,
        "num_shared_signals": 9,
        "action_dim": 7,
        "num_actions": 10,
        "action_space": "eef_delta",
        "rope_wavelength": 32.0,
        "rope_max_positions": 100,
        "sinusoidal_dim": 256,
        "flow_time_max_period": 10_000.0,
        "flow_time_frequency_scale": 1000.0,
        "frequency_max_period": 1000.0,
        "control_frequency": 3.0,
        "num_sampling_steps": 4,
    },
}


def to_config(model_name):
    return FlowerArgs(**CONFIGS[model_name])
