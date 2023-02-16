import tensorflow as tf
from tensorflow.keras.layers import Layer


class GetDropConnect(Layer):
    """Dropout for model layers.
    DropConnect is similar to dropout, but instead of setting
    activations to zero, it sets a fraction of the weights in a layer to
    zero. This helps to prevent overfitting by reducing the complexity
    of the model and encouraging the model to rely on a more diverse set
    of weights.

    # Arguments
        survival_rate: Float, survival probability to drop features.

    # Properties
        survival_rate: Float.

    # Methods
        call()

    # References
        [Deep Networks with Stochastic Depth]
        (https://arxiv.org/pdf/1603.09382.pdf)
    """
    def __init__(self, survival_rate, **kwargs):
        super(GetDropConnect, self).__init__(**kwargs)
        self.survival_rate = survival_rate

    def call(self, features, training=None):
        if training:
            batch_size = tf.shape(features)[0]
            random_tensor = self.survival_rate
            kwargs = {"shape": [batch_size, 1, 1, 1], "dtype": features.dtype}
            random_tensor = random_tensor + tf.random.uniform(**kwargs)
            binary_tensor = tf.floor(random_tensor)
            output = (features / self.survival_rate) * binary_tensor
            return output
        else:
            return features


class FuseFeature(Layer):
    """Fuse features from different resolutions and return a
    weighted sum. The resulting weighted sum is the fused feature.
    Lower layers of the network tend to extract more basic features,
    such as edges and shapes, while higher layers extract more complex
    features that are useful for making predictions.
    This class implements function that combines features from various
    levels/layers of the model. This helps to combine the strengths of
    different features and create a more robust and accurate
    representation of the input image.

    # Arguments
        fusion: Str, feature fusion method.

    # Properties
        fusion: Str.

    # Methods
        build()
        call()
        _fuse_fast()
        _fuse_sum()
        get_config()

    # References
        [EfficientDet: Scalable and Efficient Object Detection]
        (https://arxiv.org/pdf/1911.09070.pdf)
    """
    def __init__(self, fusion, **kwargs):
        super().__init__(**kwargs)
        self.fusion = fusion
        if fusion == 'fast':
            self.fuse_method = self._fuse_fast
        elif fusion == 'sum':
            self.fuse_method = self._fuse_sum
        else:
            raise ValueError('FPN weight fusion is not defined')

    def build(self, input_shape):
        num_in = len(input_shape)
        args = (self.name, (num_in,), tf.float32,
                tf.keras.initializers.constant(1 / num_in))
        self.w = self.add_weight(*args, trainable=True)

    def call(self, inputs, fusion):
        inputs = [input for input in inputs if input is not None]
        return self.fuse_method(inputs)

    def _fuse_fast(self, inputs):
        w = tf.keras.activations.relu(self.w)

        pre_activations = []
        for input_arg in range(len(inputs)):
            pre_activations.append(w[input_arg] * inputs[input_arg])
        x = tf.reduce_sum(pre_activations, 0)
        x = x / (tf.reduce_sum(w) + 0.0001)
        return x

    def _fuse_sum(self, inputs):
        x = inputs[0]
        for node in inputs[1:]:
            x = x + node
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({'fusion': self.fusion})
        return config
