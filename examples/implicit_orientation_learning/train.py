import os
import glob
import json
import argparse

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from paz.backend.image import write_image
from paz.abstract import GeneratingSequence
from paz.optimization.callbacks import DrawInferences
from paz.pipelines import AutoEncoderPredictor

from scenes import SingleView

from pipelines import DomainRandomization
from model import AutoEncoder

description = 'Training script for learning implicit orientation vector'
root_path = os.path.join(os.path.expanduser('~'), '.keras/paz/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', type=str, help='Path of 3D OBJ model',
                    default=os.path.join(
                        root_path,
                        'datasets/ycb/models/035_power_drill/textured.obj'))
parser.add_argument('-cl', '--class_name', default='035_power_drill', type=str,
                    help='Class name to be added to model save path')
parser.add_argument('-id', '--images_directory', type=str,
                    help='Path to directory containing background images',
                    default=os.path.join(
                        root_path, 'datasets/voc-backgrounds/'))
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
parser.add_argument('-sh', '--top_only', default=0, choices=[0, 1], type=int,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-ls', '--loss', default='binary_crossentropy', type=str,
                    help='tf.keras loss function name to be used')
parser.add_argument('-r', '--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
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
renderer = SingleView(args.obj_path, (args.image_size, args.image_size),
                      args.y_fov, args.depth, args.light, bool(args.top_only),
                      args.roll, args.shift)

# creating sequencer
image_paths = glob.glob(os.path.join(args.images_directory, '*.png'))
processor = DomainRandomization(
    renderer, args.image_size, image_paths, args.num_occlusions)

sequence = GeneratingSequence(processor, args.batch_size, args.steps_per_epoch)

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
save = ModelCheckpoint(
    model_path, 'loss', verbose=1, save_best_only=True, save_weights_only=True)

# setting drawing callbacks
images = (sequence.__getitem__(0)[0]['input_image'] * 255).astype('uint8')
for arg, image in enumerate(images):
    image_name = 'image_%03d.png' % arg
    image_path = os.path.join(save_path, 'original_images/' + image_name)
    write_image(image_path, image)
inferencer = AutoEncoderPredictor(model)
draw = DrawInferences(save_path, images, inferencer)

# saving hyper-parameters and model summary as text files
print(save_path)
with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(save_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

# model optimization
model.fit_generator(
    sequence,
    steps_per_epoch=args.steps_per_epoch,
    epochs=args.max_num_epochs,
    callbacks=[stop, log, save, plateau, draw],
    verbose=1,
    workers=0)
