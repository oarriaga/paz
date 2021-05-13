import tensorflow as tf


def get_activation_fn(features, act_type):
    """Apply non-linear activation function to features provided."""
    if act_type in ('silu', 'swish'):
        return tf.nn.swish(features)
    elif act_type == 'relu':
        return tf.nn.relu(features)
    else:
        raise ValueError('Unsupported act_type {}'.format(act_type))


def get_drop_connect(features, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # Deep Networks with Stochastic Depth, https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return features
    batch_size = tf.shape(features)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1],
                                       dtype=features.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = features / survival_prob * binary_tensor
    return output
