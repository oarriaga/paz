import jax
import paz
import jax.numpy as jp
import optax

import plotter
from loader import generate_data

key = jax.random.PRNGKey(777)
step_size = 0.1
num_steps = 100
DOF = 2.0
scale = 0.2

key, data_key = jax.random.split(key)
data, true_normal, true_offset, inliers, outliers = generate_data(
    200, 30, data_key
)
true_centroid = jp.array([100.0, -150, 200.0])
data = data + true_centroid
inliers = inliers + true_centroid
outliers = outliers + true_centroid
optimizer = optax.adam(step_size)
loss = paz.lock(paz.plane.student_t_loss, scale, DOF, data)
_fit = paz.lock(paz.plane.fit, optimizer, loss, num_steps)
(pred_normal, pred_centroid), losses = paz.time(_fit)(key, data)
pred_offset = -jp.dot(pred_normal, pred_centroid)

print("\n" + "=" * 20 + " RESULTS " + "=" * 20)
print(f"True Normal:    {true_normal}")
print(f"Fitted Normal:  {pred_normal}")
print(f"True Offset:    {true_offset:.4f}")
print(f"Fitted Offset:  {pred_offset:.4f}")
print(f"True Offset:    {true_centroid}")
print(f"Fitted Offset:  {pred_centroid}")
print("=" * 50 + "\n")

plotter.plot_results(inliers, outliers, pred_normal, pred_offset)
