from keras import ops
import keras


@keras.saving.register_keras_serializable("layers")
class ComputePrototypes(keras.Layer):
    def __init__(self, axis=1, **kwargs):
        super(ComputePrototypes, self).__init__(**kwargs)
        self.axis = axis

    def call(self, z_support):
        class_prototypes = ops.mean(z_support, axis=self.axis)
        return class_prototypes

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
