import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import (
    InputLayer, Input, Dense, Flatten, BatchNormalization,
    ReLU, Conv2D, MaxPool2D)


def conv_block(x, with_pool=True):
    x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2))(x)
    return x


def CONVNET(num_classes, image_shape, num_blocks=4):
    x = inputs = Input(image_shape)
    for _ in range(num_blocks):
        x = conv_block(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='CONVNET')


def MLP(hidden_size=40):
    inputs = Input((1,), name='inputs')
    x = Dense(hidden_size, activation='relu')(inputs)
    x = Dense(hidden_size, activation='relu')(x)
    outputs = Dense(1, name='outputs')(x)
    return Model(inputs=inputs, outputs=outputs, name='MLP')


def copy_model(model):
    meta_weights = model.get_weights()
    copied_model = clone_model(model)
    copied_model.set_weights(meta_weights)
    return copied_model


def gradient_step(learning_rate, gradients, parameters_old):
    return tf.subtract(parameters_old, tf.multiply(learning_rate, gradients))


def update_dense(meta_layer, learning_rate, gradients, weights_args):
    kernel_old = meta_layer.kernel
    biases_old = meta_layer.bias
    kernel_arg, biases_arg = weights_args
    kernel_grads = gradients[kernel_arg]
    biases_grads = gradients[biases_arg]
    kernel_new = gradient_step(learning_rate, kernel_grads, kernel_old)
    biases_new = gradient_step(learning_rate, biases_grads, biases_old)
    return kernel_new, biases_new


def update_batchnorm(meta_layer, learning_rate, gradients, weights_args):
    gamma_old = meta_layer.gamma
    betta_old = meta_layer.beta
    gamma_arg, betta_arg = weights_args
    gamma_grads = gradients[gamma_arg]
    betta_grads = gradients[betta_arg]
    betta_new = gradient_step(learning_rate, gamma_grads, gamma_old)
    gamma_new = gradient_step(learning_rate, betta_grads, betta_old)
    return betta_new, gamma_new


def meta_to_task(meta_model, support_gradients, learning_rate):
    layer_arg_to_weights_arg = build_layer_to_weight(meta_model)
    task_model = copy_model(meta_model)
    for layer_arg, layer in enumerate(meta_model.layers):
        if isinstance(layer, Dense):
            weights_arg = layer_arg_to_weights_arg[layer_arg]
            kernel_new, biases_new = update_dense(
                layer, learning_rate, support_gradients, weights_arg)
            task_model.layers[layer_arg].kernel = kernel_new
            task_model.layers[layer_arg].bias = biases_new
        if isinstance(layer, Conv2D):
            weights_arg = layer_arg_to_weights_arg[layer_arg]
            kernel_new, biases_new = update_dense(
                layer, learning_rate, support_gradients, weights_arg)
            task_model.layers[layer_arg].kernel = kernel_new
            task_model.layers[layer_arg].bias = biases_new
        if isinstance(layer, BatchNormalization):
            weights_arg = layer_arg_to_weights_arg[layer_arg]
            gamma_new, betta_new = update_batchnorm(
                layer, learning_rate, support_gradients, weights_arg)
            task_model.layers[layer_arg].gamma = gamma_new
            task_model.layers[layer_arg].beta = betta_new
    return task_model


def build_layer_to_weight(model):
    layer_to_weight = {}
    weights_arg = 0
    for layer_arg, layer in enumerate(model.layers):
        if isinstance(layer, InputLayer):
            continue
        elif isinstance(layer, Flatten):
            continue
        elif isinstance(layer, MaxPool2D):
            continue
        elif isinstance(layer, ReLU):
            continue
        elif isinstance(layer, BatchNormalization):
            layer_to_weight[layer_arg] = (weights_arg, weights_arg + 1)
            weights_arg = weights_arg + 2
        elif isinstance(layer, Dense):
            layer_to_weight[layer_arg] = (weights_arg, weights_arg + 1)
            weights_arg = weights_arg + 2
        elif isinstance(layer, Conv2D):
            layer_to_weight[layer_arg] = (weights_arg, weights_arg + 1)
            weights_arg = weights_arg + 2
        else:
            raise ValueError(f'Layer {layer} not supported')
    return layer_to_weight


def MAML(meta_model, compute_loss, optimizer, learning_rate=0.01):
    def fit(RNG, sampler, num_steps):
        losses = []
        total_loss = 0
        for step in range(num_steps):
            ((x_true_support, y_true_support),
             (x_true_queries, y_true_queries)) = sampler()
            x_true_support = tf.convert_to_tensor(x_true_support)
            y_true_support = tf.convert_to_tensor(y_true_support)
            x_true_queries = tf.convert_to_tensor(x_true_queries)
            y_true_queries = tf.convert_to_tensor(y_true_queries)
            with tf.GradientTape() as meta_tape:
                with tf.GradientTape() as task_tape:
                    y_pred = meta_model(x_true_support, training=True)
                    support_loss = compute_loss(y_true_support, y_pred)
                support_gradients = task_tape.gradient(
                    support_loss, meta_model.trainable_variables)
                task_model = meta_to_task(
                    meta_model, support_gradients, learning_rate)
                y_task_pred = task_model(x_true_queries, training=True)
                task_loss = compute_loss(y_true_queries, y_task_pred)
            meta_weights = meta_model.trainable_variables
            gradients = meta_tape.gradient(task_loss, meta_weights)
            optimizer.apply_gradients(zip(gradients, meta_weights))
            total_loss = total_loss + task_loss
            print('step {} | loss = {}'.format(step, total_loss / (step + 1)))
        return losses
    return fit


def Predict(model, learning_rate, compute_loss):
    def call(x_support, y_support, x_queries, num_steps):
        model_copy = copy_model(model)
        model_copy.compile(SGD(learning_rate), compute_loss)
        for step in range(num_steps):
            model_copy.fit(x_support, y_support, epochs=1, verbose=0)
        y_queries_pred = model_copy(x_queries)
        return y_queries_pred
    return call


def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
