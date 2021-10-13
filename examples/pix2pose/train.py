import os
import glob
import argparse
import numpy as np
import time

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from paz.abstract import GeneratingSequence
from paz.abstract.sequence import GeneratingSequence

from pipelines import GeneratingSequencePix2Pose, GeneratedImageGenerator, make_batch_discriminator
from model import Generator, Discriminator, loss_color_wrapped, loss_error


description = 'Training script Pix2Pose model'
root_path = os.path.join(os.path.expanduser('~'), '.keras/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-cl', '--class_name', default='tless05', type=str,
                    help='Class name to be added to model save path')
parser.add_argument('-id', '--background_images_directory', type=str,
                    help='Path to directory containing background images')
parser.add_argument('-pi', '--images_directory', type=str,
                    help='Path to pre-generated images (npy format)')
parser.add_argument('-bs', '--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-ld', '--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-e', '--max_num_epochs', default=10000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-st', '--steps_per_epoch', default=5, type=int,
                    help='Steps per epoch')
parser.add_argument('-oc', '--num_occlusions', default=2, type=int,
                    help='Number of occlusions')
parser.add_argument('-sa', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-rm', '--rotation_matrices',
                    type=str, help='Path to npy file with a list of rotation matrices', required=True)
parser.add_argument('-de', '--description',
                    type=str, help='Description of the model')
args = parser.parse_args()

# Building the whole GAN model
dcgan_input = Input(shape=(128, 128, 3))
discriminator = Discriminator()
generator = Generator()
color_output, error_output = generator(dcgan_input)
discriminator.trainable = False
discriminator_output = discriminator(color_output)
dcgan = Model(inputs=[dcgan_input], outputs={"color_output": color_output, "error_output": error_output, "discriminator_output": discriminator_output})

# For the loss function pix2pose needs to know all the rotations under which the pose looks the same
rotation_matrices = np.load(args.rotation_matrices)
loss_color = loss_color_wrapped(rotation_matrices)

# Set the loss
optimizer = Adam(args.learning_rate, amsgrad=True)
losses = {"color_output": loss_color,
          "error_output": loss_error,
          "discriminator_output": "binary_crossentropy"}
lossWeights = {"color_output": 100.0, "error_output": 50.0, "discriminator_output": 1.0}
dcgan.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights, run_eagerly=True)

discriminator.trainable = True
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer)

# Creating sequencer
background_image_paths = glob.glob(os.path.join(args.background_images_directory, '*.jpg'))
processor_train = GeneratedImageGenerator(os.path.join(args.images_directory, "train"), args.image_size, background_image_paths, num_occlusions=0)
processor_test = GeneratedImageGenerator(os.path.join(args.images_directory, "test"), args.image_size, background_image_paths, num_occlusions=0)
sequence_train = GeneratingSequencePix2Pose(processor_train, dcgan, args.batch_size, args.steps_per_epoch, rotation_matrices=rotation_matrices)
sequence_test = GeneratingSequencePix2Pose(processor_test, dcgan, args.batch_size, args.steps_per_epoch, rotation_matrices=rotation_matrices)

# Making directory for saving model weights and logs
model_name = '_'.join([dcgan.name, args.class_name])
save_path = os.path.join(args.save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Setting callbacks
log = CSVLogger(os.path.join(save_path, '%s.log' % model_name))
log.model = dcgan

callbacks=[log]

for callback in callbacks:
    callback.on_train_begin()

for num_epoch in range(args.max_num_epochs):
    sequence_iterator_train = sequence_train.__iter__()
    sequence_iterator_test = sequence_test.__iter__()

    for callback in callbacks:
        callback.on_epoch_begin(num_epoch)

    for num_batch in range(args.steps_per_epoch):
        # Train the discriminator
        discriminator.trainable = True
        batch = next(sequence_iterator_train)

        X_discriminator_real, y_discriminator_real = make_batch_discriminator(generator, batch[0]['input_image'], batch[1]['color_output'], 1)
        loss_discriminator_real = discriminator.train_on_batch(X_discriminator_real, y_discriminator_real)

        X_discriminator_fake, y_discriminator_fake = make_batch_discriminator(generator, batch[0]['input_image'], batch[1]['color_output'], 0)
        loss_discriminator_fake = discriminator.train_on_batch(X_discriminator_fake, y_discriminator_fake)

        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake)/2.

        # Train the generator
        discriminator.trainable = False
        loss_dcgan, loss_color_output, loss_dcgan_discriminator, loss_error_output = dcgan.train_on_batch(batch[0]['input_image'], {"color_output": batch[1]['color_output'], "error_output": batch[1]['error_output'], "discriminator_output": np.ones((args.batch_size, 1))})

        # Test the network
        batch_test = next(sequence_iterator_test)
        loss_dcgan_test, loss_color_output_test, loss_dcgan_discriminator_test, loss_error_output_test = dcgan.test_on_batch(batch_test[0]['input_image'], {"color_output": batch_test[1]['color_output'], "error_output": batch_test[1]['error_output'], "discriminator_output": np.ones((args.batch_size, 1))})

        print("Loss DCGAN: {}".format(loss_dcgan))
    for callback in callbacks:
        callback.on_epoch_end(num_epoch, logs={'loss_discriminator': loss_discriminator,
                                               'loss_dcgan': loss_dcgan, 'loss_color_output': loss_color_output,
                                               'loss_dcgan_discriminator': loss_dcgan_discriminator,
                                               'loss_error_output': loss_error_output,
                                               'loss_dcgan_test': loss_dcgan_test, 'loss_color_output_test': loss_color_output_test,
                                               'loss_dcgan_discriminator_test': loss_dcgan_discriminator_test,
                                               'loss_error_output_test': loss_error_output_test
                                               })


for callback in callbacks:
    callback.on_train_end()