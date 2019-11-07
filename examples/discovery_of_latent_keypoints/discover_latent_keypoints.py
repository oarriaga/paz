import os
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["PYOPENGL_PLATFORM"] = 'egl'

import json
import argparse
from paz.models import KeypointNetShared
from paz.models import Projector
from paz.pipelines import KeypointInference
from paz.pipelines import KeypointSharedAugmentation
from paz.core.sequencer import GeneratingSequencer
from paz.optimization.callbacks import DrawInferences
from paz.optimization import KeypointNetLoss

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Progbar

from poseur.scenes import MultiView
import numpy as np

description = 'Training script for learning latent 3D keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-cl', '--class_name', default='035_power_drill', type=str,
                    help='``FERPlus``, ``FER`` or ``IMDB``')
parser.add_argument('-nk', '--num_keypoints', default=10, type=int,
                    help='Number of keypoints to be learned')
parser.add_argument('-f', '--filters', default=64, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-bs', '--batch_size', default=5, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-is', '--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-sp', '--stop_patience', default=7, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('-pp', '--plateau_patience', default=3, type=int,
                    help='Number of epochs before reducing learning rate')
parser.add_argument('-e', '--max_num_epochs', default=2, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-st', '--steps_per_epoch', default=3, type=int,
                    help='Steps per epoch')
parser.add_argument('-sa', '--save_path', default='trained_models/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-sh', '--sphere', default='full',
                    choices=['full', 'half'], type=str,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-r', '--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
parser.add_argument('-t', '--translation', default=None, type=float,
                    help='Threshold for translation')
parser.add_argument('-d', '--depth', default=0.30, type=float,
                    help='Distance from camera to origin in meters')
parser.add_argument('-s', '--shift', default=0.05, type=float,
                    help='Threshold of random shift of camera')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('-l', '--light', default=5.0, type=float,
                    help='Light intensity from poseur')
parser.add_argument('-bk', '--background', default=0, type=int,
                    help='Background color')
parser.add_argument('-ap', '--alpha', default=0.1, type=float,
                    help='Alpha leaky-relu parameter')
args = parser.parse_args()


class_name = '035_power_drill'
OBJ_filepath = '.keras/altamira/datasets/models/%s/textured.obj' % class_name
OBJ_filepath = os.path.join(os.path.expanduser('~'), OBJ_filepath)
save_path = os.path.join(os.path.expanduser('~'), '.keras/paz/models/')

# setting scene
scene = MultiView(OBJ_filepath, (args.image_size, args.image_size),
                  args.y_fov, args.depth, args.sphere, args.roll,
                  args.translation, args.shift, args.light, args.background)
focal_length = scene.camera.get_projection_matrix()[0, 0]

# setting sequencer
input_shape = (args.image_size, args.image_size, 3)
projector = Projector(focal_length, True)
processor = KeypointSharedAugmentation(scene, projector, args.image_size)
sequencer = GeneratingSequencer(processor, args.batch_size, as_list=True)

# model instantiation
model = KeypointNetShared(input_shape, args.num_keypoints,
                          args.depth * 10, args.filters, args.alpha)

# loss instantiation
loss = KeypointNetLoss(args.num_keypoints, focal_length)
losses = {'uvz_points-shared': loss.uvz_points,
          'uv_volumes-shared': loss.uv_volumes}
uvz_point_losses = [loss.consistency, loss.separation, loss.relative_pose]

# metrics
metrics = {'uvz_points-shared': uvz_point_losses,
           'uv_volumes-shared': [loss.silhouette, loss.variance]}

# model compilation
optimizer = Adam(args.learning_rate, amsgrad=True)
model.compile(optimizer, losses, metrics)
model_name = '_'.join([model.name, str(args.num_keypoints), class_name])
model.summary()

# making directory for saving model weights and logs
save_path = os.path.join(save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# setting callbacks
log = CSVLogger(os.path.join(save_path, '%s.log' % model_name))
stop = EarlyStopping('loss', patience=args.stop_patience, verbose=1)
plateau = ReduceLROnPlateau('loss', patience=args.plateau_patience, verbose=1)
inferencer = KeypointInference(
    model.get_layer('keypointnet'), args.num_keypoints, to_BGR=True)
images = (sequencer.__getitem__(0)[0][0] * 255).astype('uint8')
draw = DrawInferences(save_path, images, inferencer)
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
save = ModelCheckpoint(model_path, 'loss', verbose=1,
                       save_best_only=True, save_weights_only=True)

# saving hyper-parameters and model summary as text files
with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(save_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

# model optimization
model.fit_generator(
    sequencer,
    steps_per_epoch=args.steps_per_epoch,
    epochs=args.max_num_epochs,
    callbacks=[stop, log, save, plateau, draw],
    verbose=1,
    workers=0)

# re-loading best model
model.load_weights(model_path)
model = model.get_layer('keypointnet')

# rendering multiple scenes for forward passing
keypoints_set, projector = [], Projector(focal_length, True)
print('Calculating mean and variance of discovered keypoints...')
num_forward_passes = 1000  # number of samples for calculating the mean
sequencer.batch_size = 1   # changing batch size of sequencer
progress_bar = Progbar(num_forward_passes)
for batch_arg in range(num_forward_passes):
    (image_A, image_B), (matrices, labels) = sequencer.__getitem__(batch_arg)
    matrices = matrices[0].reshape(4, 4, 4)
    world_to_A, A_to_world = matrices[0], matrices[2]
    keypoints = model.predict(image_A)[0]
    keypoints3D = projector.unproject(keypoints)
    keypoints3D = np.squeeze(keypoints3D)
    keypoints3D_in_world = np.matmul(keypoints3D, A_to_world.T)
    keypoints_set.append(np.expand_dims(keypoints3D_in_world, 0))
    progress_bar.update(batch_arg + 1)

# calculating mean and variance and writing it in ``save_path``
keypoints_sets = np.concatenate(keypoints_set, axis=0)
mean = np.mean(keypoints_sets, axis=0)
variance = np.var(keypoints_sets, axis=0)
print('keypoints mean: \n', mean)
print('keypoints variance: \n', variance)
np.savetxt(os.path.join(save_path, 'keypoints_mean.txt'), mean)
np.savetxt(os.path.join(save_path, 'keypoints_variance.txt'), variance)
