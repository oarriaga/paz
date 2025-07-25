import os
from functools import partial

os.environ["KERAS_BACKEND"] = "jax"

import jax
import paz

import tensorflow as tf  # We import TF so we can use tf.data.
import keras
import numpy as np


def MNIST(batch_size=32):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784)).astype("float32")
    x_test = np.reshape(x_test, (-1, 784)).astype("float32")
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    return train_dataset, val_dataset


def Model(input_shape):
    inputs = keras.Input(input_shape, name="digits")
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(10, name="predictions")(x)
    return keras.Model(inputs, outputs)


def compute_loss_and_updates(theta, theta_static, x, y, model, compute_loss):
    args = (theta, theta_static, x)
    y_pred, theta_static = model.stateless_call(*args, training=True)
    loss = compute_loss(y, y_pred)
    return loss, theta_static


def optimize_step(state, data, optimizer, compute_gradients):
    theta, theta_static, optimizer_theta = state
    x, y = data
    (loss, theta_static), grads = compute_gradients(theta, theta_static, x, y)
    args = (optimizer_theta, grads, theta)
    theta, optimizer_theta = optimizer.stateless_apply(*args)
    return loss, (theta, theta_static, optimizer_theta)


batch_size = 32
input_shape = (784,)
model = Model(input_shape)
compute_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
compute_loss = paz.lock(compute_loss_and_updates, model, compute_loss)
compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)
optimizer.build(model.trainable_variables)
state = (
    model.trainable_variables,
    model.non_trainable_variables,
    optimizer.variables,
)
_optimize_step = jax.jit(paz.lock(optimize_step, optimizer, compute_gradients))

train_dataset, _ = MNIST(batch_size)

for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = _optimize_step(state, data)
    if step % 100 == 0:
        print(f"Training batch loss at step {step}: {float(loss):.4f}")


theta, theta_static, optimizer_variables = state
for variable, value in zip(model.trainable_variables, theta):
    variable.assign(value)
for variable, value in zip(model.non_trainable_variables, theta_static):
    variable.assign(value)
