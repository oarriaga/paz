import h5py
import numpy as np
from paz.core import ops

from mnist import MNIST


top_k = 10
x_test, y_test = MNIST('test').load()
x_test, y_test = x_test, y_test

metrics = h5py.File('data/metrics.hdf5')
metrics = np.asarray(metrics['metrics'])[..., 0]
loss = np.sum(metrics, axis=0)
loss_args = np.argsort(loss)

print('Displaying easiest samples')
for arg in loss_args[:top_k]:
    ops.show_image((x_test[arg] * 255).astype('uint8'))

print('Displaying most difficult samples')
for arg in loss_args[::-1][:top_k]:
    ops.show_image((x_test[arg] * 255).astype('uint8'))
