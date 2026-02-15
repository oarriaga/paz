import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["KERAS_BACKEND"] = "jax"
import jax.numpy as jp
import paz
import jax
import matplotlib.pyplot as plt

key = jax.random.key(777)
data = jax.random.normal(key, (1000, 2))
data = (jp.array([[2.0, 1.0], [1.0, 3.0]]) @ data.T).T

state = paz.PCA.fit(data, 2)

figure, axis = plt.subplots(figsize=(8, 6))

axis.scatter(data[:, 0], data[:, 1])
axis.set_xlim((-10, 10))
axis.set_ylim((-10, 10))

axis.quiver(
    state.mean[0],
    state.mean[1],
    state.base[0, 0],
    state.base[1, 0],
    scale=3,
    color="red",
    label="Principal Component 1",
)

axis.quiver(
    state.mean[0],
    state.mean[1],
    state.base[0, 1],
    state.base[1, 1],
    scale=3,
    color="green",
    label="Principal Component 2",
)

axis.set_xlabel("Feature 1")
axis.set_ylabel("Feature 2")
axis.set_title("Original Data with Principal Components")
axis.legend()
plt.show()
