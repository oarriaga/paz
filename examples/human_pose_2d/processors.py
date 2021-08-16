import numpy as np
import tensorflow as tf
from paz import processors as pr


class LoadModel(pr.Processor):
    def __init__(self):
        super(LoadModel, self).__init__()

    def call(self, model_path):
        return tf.keras.models.load_model(model_path)


class CreateDirectory(pr.Processor):
    def __init__(self):
        super(CreateDirectory, self).__init__()

    def call(self, directory):
        print('=> creating {}'.format(directory))
        directory.mkdir(parents=True, exist_ok=True)


class ReplaceText(pr.Processor):
    def __init__(self, text):
        super(ReplaceText, self).__init__()
        self.text = text

    def call(self, oldvalue, newvalue):
        return self.text.replace(oldvalue, newvalue)


class MaxPooling2D(pr.Processor):
    def __init__(self):
        super(MaxPooling2D, self).__init__()

    def call(self, pool_size, strides, padding):
        pool = tf.keras.layers.MaxPooling2D(pool_size, strides, padding)
        return pool


class Transpose(pr.Processor):
    def __init__(self):
        super(Transpose, self).__init__()

    def call(self, tensor, permutes):
        return tf.transpose(tensor, permutes)


class CompareElementWiseEquality(pr.processor):
    def __init__(self):
        super(CompareElementWiseEquality, self).__init__()

    def call(self, tensor_1, tensor_2):
        return tf.math.equal(tensor_1, tensor_2)


class ChangeDataType(pr.processor):
    def __init__(self):
        super(ChangeDataType, self).__init__()

    def call(self, tensor, dtype):
        return tf.cast(tensor, dtype)


class MultiplyTensors(pr.Processor):
    def __init__(self):
        super(MultiplyTensors, self).__init__()

    def call(self, tensor_1, tensor_2):
        return tensor_1 * tensor_2
        # return tf.math.multiply(tensor_1, tensor_2)