import os
import glob
from tensorflow.keras.optimizers import Adam
from paz.abstract import GeneratingSequence
from paz.models.segmentation import UNET_VGG16
from models.generator import Generator
from models.discriminator import Discriminator
from models.pix2pose import Pix2Pose
from tensorflow.keras.losses import BinaryCrossentropy
# from paz.backend.image import show_image, resize_image
# import numpy as np

from scenes import PixelMaskRenderer
from pipelines import DomainRandomization
from loss import WeightedReconstruction
from loss import WeightedReconstructionWithError
from loss import ErrorPrediction
# from metrics import error_prediction, weighted_reconstruction
# from metrics import weighted_reconstruction_with_error
from metrics import mean_squared_error, error_prediction
from metrics import weighted_reconstruction_wrapper
# from models.fully_convolutional_net import FullyConvolutionalNet

H, W, num_channels = image_shape = [128, 128, 3]
root_path = os.path.expanduser('~')
background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)
path_OBJ = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
path_OBJ = os.path.join(root_path, path_OBJ)
num_occlusions = 1
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
distance = [0.3, 0.5]
light = [1.0, 30]
top_only = False
roll = 3.14159
shift = 0.05
num_steps = 1000
batch_size = 32
beta = 3.0
alpha = 0.1
filters = 16
num_classes = 3
learning_rate = 0.001
# steps_per_epoch
model_names = ['PIX2POSE', 'PIX2POSE_GENERATOR', 'UNET_VGG16']
# model_name = 'UNET_VGG16'
# model_name = 'PIX2POSE_GENERATOR'
model_name = 'PIX2POSE'
max_num_epochs = 1
latent_dimension = 128
beta = 3.0


renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)


# model = FullyConvolutionalNet(num_classes, image_shape, filters, alpha)
# name_to_model = dict(zip(model_names, [Generator, UNET_VGG16])
# model = name_to_model[model_name]

if model_name == 'UNET_VGG16':
    model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
    loss = WeightedReconstruction(beta)
    inputs_to_shape = {'input_1': [H, W, num_channels]}
    labels_to_shape = {'masks': [H, W, 4]}
    weighted_reconstruction = weighted_reconstruction_wrapper(beta, False)
    metrics = {'masks': [weighted_reconstruction, mean_squared_error]}
    optimizer = Adam(learning_rate)
    model.compile(optimizer, loss, metrics)

# TODO this is not working at the moment because the loss does not include 
# the error prediction loss.
if model_name == 'PIX2POSE_GENERATOR':
    model = Generator(image_shape, latent_dimension)
    reconstruction_loss = WeightedReconstructionWithError(beta)
    loss = WeightedReconstructionWithError()
    H, W, num_channels = image_shape
    inputs_to_shape = {'RGB_input': [H, W, num_channels]}
    labels_to_shape = {'RGB_with_error': [H, W, 4]}
    weighted_reconstruction = weighted_reconstruction_wrapper(beta, True)
    metrics = {'RGB_with_error':
               [weighted_reconstruction, error_prediction, mean_squared_error]}
    optimizer = Adam(learning_rate)
    model.compile(optimizer, loss, metrics)

if model_name == 'PIX2POSE':
    discriminator = Discriminator(image_shape)
    generator = Generator(image_shape, latent_dimension)
    model = Pix2Pose(image_shape, discriminator, generator, latent_dimension)
    H, W, num_channels = image_shape
    inputs_to_shape = {'RGB_input': [H, W, num_channels]}
    labels_to_shape = {'RGB_with_error': [H, W, 4]}
    optimizers = {'discriminator': Adam(learning_rate),
                  'generator': Adam(learning_rate)}
    losses = {'discriminator': BinaryCrossentropy(),
              'weighted_reconstruction': WeightedReconstructionWithError(),
              'error_prediction': ErrorPrediction()}
    loss_weights = {'weighted_reconstruction': 100, 'error_prediction': 50}
    model.compile(optimizers, losses, loss_weights)

processor = DomainRandomization(
    renderer, image_shape, image_paths, inputs_to_shape,
    labels_to_shape, num_occlusions)

sequence = GeneratingSequence(processor, batch_size, num_steps)
model.load_weights('PIX2POSE_GAN.hdf5')
"""
model.fit(
    sequence,
    epochs=max_num_epochs,
    # callbacks=[stop, log, save, plateau, draw],
    verbose=1,
    workers=0)

model.save_weights('PIX2POSE_GAN.hdf5')
"""
"""
def normalize(image):
    return (image * 255.0).astype('uint8')


def show_results():
    # image, alpha, pixel_mask_true = renderer.render()
    sample = processor()
    image = sample['inputs']['input_1']
    pixel_mask_true = sample['labels']['masks']
    image = np.expand_dims(image, 0)
    pixel_mask_pred = model.predict(image)
    pixel_mask_pred = normalize(np.squeeze(pixel_mask_pred, axis=0))
    image = normalize(np.squeeze(image, axis=0))
    results = np.concatenate(
        [image, normalize(pixel_mask_true[..., 0:3]), pixel_mask_pred], axis=1)
    H, W = results.shape[:2]
    scale = 6
    results = resize_image(results, (scale * W, scale * H))
    show_image(results)
"""
