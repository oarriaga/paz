import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import json
import glob
import argparse

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from paz.core.sequencer import GeneratingSequencer
from paz.optimization.callbacks import DrawInferences

from poseur.scenes import SingleView

from pipelines import SelfSupervisedAugmentation, AutoEncoderInference
from model import AutoEncoder
from paz.core import ops

description = 'Training script for learning 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    '-op', '--obj_path',
    default=os.path.join(
        os.path.expanduser('~'),
        '.keras/paz/datasets/ycb/models/035_power_drill/textured.obj'),
    type=str, help='Path for writing model weights and logs')
parser.add_argument('-cl', '--class_name', default='035_power_drill', type=str,
                    help='Class name to be added to model save path')
parser.add_argument('-id', '--images_directory', default=None, type=str,
                    help='Path to directory containing background images')
parser.add_argument('-bs', '--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-is', '--latent_dimension', default=128, type=int,
                    help='Latent dimension of the auto-encoder')
parser.add_argument('-ld', '--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-sp', '--stop_patience', default=7, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('-pp', '--plateau_patience', default=3, type=int,
                    help='Number of epochs before reducing learning rate')
parser.add_argument('-e', '--max_num_epochs', default=10000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-st', '--steps_per_epoch', default=1000, type=int,
                    help='Steps per epoch')
parser.add_argument('-sh', '--sphere', default='full',
                    choices=['full', 'half'], type=str,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-ls', '--loss', default='binary_crossentropy', type=str,
                    help='tf.keras loss function name to be used')
parser.add_argument('-r', '--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
parser.add_argument('-t', '--translation', default=0.05, type=float,
                    help='Threshold for translation')
parser.add_argument('-s', '--shift', default=0.05, type=float,
                    help='Threshold of random shift of camera')
parser.add_argument('-d', '--depth', nargs='+', type=float,
                    default=[0.3, 0.5],
                    help='Distance from camera to origin in meters')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('-l', '--light', nargs='+', type=float,
                    default=[.5, 30],
                    help='Light intensity from poseur')
parser.add_argument('-oc', '--num_occlusions', default=2, type=int,
                    help='Number of occlusions')
parser.add_argument('-sa', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
args = parser.parse_args()

# setting optimizer and compiling model
latent_dimension = args.latent_dimension
model = AutoEncoder((args.image_size, args.image_size, 3), latent_dimension)
optimizer = Adam(args.learning_rate, amsgrad=True)
model.compile(optimizer, args.loss, metrics=['mse'])
model.summary()

# setting scene
renderer = SingleView(
    args.obj_path, (args.image_size, args.image_size),
    args.y_fov, args.depth, args.sphere, args.roll,
    args.translation, args.shift, args.light)

# creating sequencer
image_paths = glob.glob(os.path.join(args.images_directory, '*.png'))
processor = SelfSupervisedAugmentation(
    renderer, image_paths, args.image_size, args.num_occlusions)
sequencer = GeneratingSequencer(processor, args.batch_size)

# making directory for saving model weights and logs
model_name = '_'.join([model.name, str(latent_dimension), args.class_name])
save_path = os.path.join(args.save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# setting callbacks
log = CSVLogger(os.path.join(save_path, '%s.log' % model_name))
stop = EarlyStopping('loss', patience=args.stop_patience, verbose=1)
plateau = ReduceLROnPlateau('loss', patience=args.plateau_patience, verbose=1)
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
save = ModelCheckpoint(model_path, 'loss', verbose=1,
                       save_best_only=True, save_weights_only=True)
images = (sequencer.__getitem__(0)[0]['image'] * 255).astype('uint8')

ops.save_images(os.path.join(save_path, 'original_images/'), images)
inferencer = AutoEncoderInference(model)
draw = DrawInferences(save_path, images, inferencer,
                      label_topic='reconstruction')

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
