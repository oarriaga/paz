import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf

from paz.backend.image import write_image
from paz.abstract import GeneratingSequence
from paz.abstract.sequence import GeneratingSequencePix2Pose, GeneratingSequence
from paz.optimization.callbacks import DrawInferences
from paz.pipelines import AutoEncoderPredictor

from scenes import SingleView

from pipelines import DepthImageGenerator, RendererDataGenerator, make_batch_discriminator
from model import Generator, Discriminator, transformer_loss, loss_color, loss_error, PlotImagesCallback


description = 'Training script for learning implicit orientation vector'
root_path = os.path.join(os.path.expanduser('~'), '.keras/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', type=str, help='Path of 3D OBJ model',
                    default=os.path.join(
                        root_path,
                        'datasets/035_power_drill/tsdf/textured.obj'))
parser.add_argument('-cl', '--class_name', default='035_power_drill', type=str,
                    help='Class name to be added to model save path')
parser.add_argument('-id', '--images_directory', type=str,
                    help='Path to directory containing background images',
                    default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-bs', '--batch_size', default=4, type=int,
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
parser.add_argument('-st', '--steps_per_epoch', default=5, type=int,
                    help='Steps per epoch')
parser.add_argument('-sh', '--top_only', default=0, choices=[0, 1], type=int,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-ls', '--loss', default=transformer_loss, type=str,
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

dcgan_input = Input(shape=(128, 128, 3))
discriminator = Discriminator()
generator = Generator()
color_output, error_output = generator(dcgan_input)
discriminator.trainable = False
discriminator_output = discriminator(color_output)
dcgan = Model(inputs=[dcgan_input], outputs={"color_output": color_output, "error_output": error_output, "discriminator_output": discriminator_output})
#plot_model(dcgan, to_file='model.png', show_shapes=True, show_layer_names=True)

optimizer = Adam(args.learning_rate, amsgrad=True)
losses = {"color_output": loss_color,
          "error_output": loss_error,
          "discriminator_output": "binary_crossentropy"}
lossWeights = {"color_output": 1.0, "error_output": 1.0, "discriminator_output": 1.0}
dcgan.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights, run_eagerly=True)
dcgan.summary()

discriminator.trainable = True
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer)
discriminator.summary()

# setting scene
renderer = SingleView(args.obj_path, (args.image_size, args.image_size),
                      args.y_fov, args.depth, args.light, bool(args.top_only),
                      args.roll, args.shift)

#generator = RendererDataGenerator(renderer=renderer, steps_per_epoch=args.steps_per_epoch, batch_size=128)

# creating sequencer
image_paths = glob.glob(os.path.join(args.images_directory, '*.jpg'))
print(os.path.join(args.images_directory, '*.jpg'))
print(image_paths[:10])
processor = DepthImageGenerator(renderer, args.image_size, image_paths, num_occlusions=0)
sequence = GeneratingSequencePix2Pose(processor, dcgan, args.batch_size, args.steps_per_epoch)
#sequence = GeneratingSequence(processor, args.batch_size, args.steps_per_epoch)

# making directory for saving model weights and logs
model_name = '_'.join([dcgan.name, str(latent_dimension), args.class_name])
save_path = os.path.join(args.save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# setting callbacks
log = CSVLogger(os.path.join(save_path, '%s.log' % model_name))
log.model = dcgan
stop = EarlyStopping('loss', patience=args.stop_patience, verbose=1)
stop.model = dcgan
plateau = ReduceLROnPlateau('loss', patience=args.plateau_patience, verbose=1)
plateau.model = dcgan
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
save = ModelCheckpoint(
    model_path, 'loss', verbose=1, save_best_only=True, save_weights_only=False)
save.model = dcgan
plot = PlotImagesCallback(dcgan, sequence, save_path)

callbacks=[stop, log, save, plateau, plot]

# setting drawing callbacks
#images = (sequence.__getitem__(0)[0]['input_image'] * 255).astype('uint8')
#for arg, image in enumerate(images):
#    image_name = 'image_%03d.png' % arg
#    image_path = os.path.join(save_path, 'original_images/' + image_name)
#    write_image(image_path, image)
#inferencer = AutoEncoderPredictor(model)
#draw = DrawInferences(save_path, images, inferencer)

# saving hyper-parameters and model summary as text files
args.__dict__['loss'] = args.__dict__['loss'].__name__
print(save_path)
print(args.__dict__)
with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(save_path, 'model_summary.txt'), 'w') as filer:
    dcgan.summary(print_fn=lambda x: filer.write(x + '\n'))

# Get the iterator from the sequence
#sequence_iterator = sequence.__iter__()

for callback in callbacks:
    callback.on_train_begin()

for num_epoch in range(args.max_num_epochs):
    sequence_iterator = sequence.__iter__()
    for callback in callbacks:
        callback.on_epoch_begin(num_epoch)

    for num_batch in range(args.steps_per_epoch):
        discriminator.trainable = True
        batch = next(sequence_iterator)

        X_discriminator_real, y_discriminator_real = make_batch_discriminator(generator, batch[0]['input_image'], batch[1]['color_output'], 1)
        loss_discriminator_real = discriminator.train_on_batch(X_discriminator_real, y_discriminator_real)

        X_discriminator_fake, y_discriminator_fake = make_batch_discriminator(generator, batch[0]['input_image'], batch[1]['color_output'], 0)
        loss_discriminator_fake = discriminator.train_on_batch(X_discriminator_fake, y_discriminator_fake)

        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake)/2.

        discriminator.trainable = False

        loss_dcgan = dcgan.train_on_batch(batch[0]['input_image'], {"color_output": batch[1]['color_output'], "error_output": batch[1]['error_output'], "discriminator_output": np.ones((4, 1))})

        print("Loss discriminator: {}, Loss DCGAN: {}".format(loss_discriminator, loss_dcgan))

    for callback in callbacks:
        print(callback)
        callback.on_epoch_end(num_epoch)


for callback in callbacks:
    callback.on_train_end()