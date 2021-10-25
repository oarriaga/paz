import os
import glob
from tensorflow.keras.optimizers import Adam
from paz.abstract import GeneratingSequence
from paz.models.segmentation import UNET_VGG16
from paz.backend.image import show_image, resize_image
import numpy as np

from scenes import PixelMaskRenderer
from pipelines import DomainRandomization
from loss import WeightedForeground, MSE_with_alpha_channel
from models.fully_convolutional_net import FullyConvolutionalNet

image_shape = [128, 128, 3]
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
max_num_epochs = 10
steps_per_epoch = num_steps


renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)

processor = DomainRandomization(renderer, image_shape,
                                image_paths, num_occlusions)

sequence = GeneratingSequence(processor, batch_size, num_steps)

beta = 3.0
weighted_foreground = WeightedForeground(beta)

# model = FullyConvolutionalNet(num_classes, image_shape, filters, alpha)
model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
# model.
optimizer = Adam(learning_rate)
# model.load_weights('UNET_weights_MSE.hdf5')
model.compile(optimizer, weighted_foreground, metrics=MSE_with_alpha_channel)
model.fit(
    sequence,
    # steps_per_epoch=args.steps_per_epoch,
    epochs=max_num_epochs,
    # callbacks=[stop, log, save, plateau, draw],
    verbose=1,
    workers=0)
# batch = sequence.__getitem__(0)
# for _ in range(100):
# image, alpha, RGB_mask = renderer.render()
# show_image(image)
# show_image(RGB_mask)


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
for _ in range(100):
    sample = processor()
    inputs, labels = sample['inputs'], sample['labels']
    show_image((inputs['input_image'] * 255).astype('uint8'))
    show_image((labels['label_image'] * 255).astype('uint8'))
"""
