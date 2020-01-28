import h5py
import numpy as np
from paz.core import ops

from mnist import MNIST


top_k = 10
data = MNIST('test').load()

metrics = h5py.File('evaluations.hdf5', 'r')

metrics = np.asarray(metrics['evaluations'])[..., 0]
loss = np.sum(metrics, axis=0)
loss_args = np.argsort(loss)

print('Displaying easiest samples')
for arg in loss_args[:top_k]:
    ops.show_image((data[arg]['image']).astype('uint8'))

print('Displaying most difficult samples')
for arg in loss_args[::-1][:top_k]:
    ops.show_image((data[arg]['image']).astype('uint8'))
