import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model

from config import Config
from pipeline import DetectionPipeline
from paz.models.detection.utils import create_prior_boxes
from utils import DataGenerator
from utils import display_top_masks, build_fpn_mask_graph, fpn_classifier_graph
from paz.datasets.shapes import Shapes
from model import MaskRCNN, get_imagenet_weights
import numpy as np
import cv2

from tensorflow.keras.layers import Layer, Input, Lambda
from layers import DetectionTargetLayer, ProposalLayer


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    """
    NAME = 'shapes'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    NUM_CLASSES = 4
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5

#Extra arguments to be passed to model from default values


description = 'Training script for Mask RCNN model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bs', '--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('-dp', '--data_path', default='/Users/poornimakaushik/Desktop/output/',
                    required=False,type=str, help='Directory for loading data')
parser.add_argument('-sp', '--save_path', default='/Users/poornimakaushik/Desktop/output/',
                    required=False,metavar='/path/to/save',
                    help="Path to save model weights and logs")
parser.add_argument('-lr', '--learning_rate', default=0.002, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-st', '--steps_per_epoch', default=1000, type=int,
                    help='steps per epoch for training')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-e', '--num_epochs', default=5, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-et', '--evaluation_period', default=1, type=int,
                    help='evaluation frequency')
parser.add_argument('--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-l', '--layers', default='heads', type=str,
                    help='Select which layers to train')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
args = parser.parse_args()
print('Path to save model: ', args.save_path)
print('Data path: ', args.data_path)


#Dataset initialisation
optimizer = SGD(args.learning_rate, args.momentum)
config = ShapesConfig()
dataset_train = Shapes(50, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))
dataset_val = Shapes(5, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))
train_generator = DataGenerator(dataset_train, config, shuffle=True,
                                 augmentation=None, detection_targets=True)
val_generator = DataGenerator(dataset_val, config, shuffle=True, detection_targets=True)

#Initial model description
model = MaskRCNN(config=config, model_dir=args.data_path, train_bn=config.TRAIN_BN,
                 image_shape=config.IMAGE_SHAPE, backbone=config.BACKBONE,
                 top_down_pyramid_size=config.TOP_DOWN_PYRAMID_SIZE)

#Network head creation
losses = model.build_complete_network()

model.keras_model.load_weights('weights/mask_rcnn_coco (1).h5',by_name=True, skip_mismatch=True)

#Set which layers to train in the backbone Default: Heads
layer_regex = {
    # all layers but the backbone
    'heads': r'(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    # From a specific Resnet stage and up
    '3+': r'(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    '4+': r'(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    '5+': r'(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
    # All layers
    'all': '.*',
}

layers=[]
if args.layers in layer_regex.keys():
    layers = layer_regex[args.layers]
model.set_trainable(layer_regex=layers)

reg_losses = [
    l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
    for w in model.keras_model.trainable_weights
    if 'gamma' not in w.name and 'beta' not in w.name]

model.keras_model.add_loss(tf.add_n(reg_losses))

for loss in losses:
    model.keras_model.add_loss(loss)

loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

for name in loss_names:
    if name in model.keras_model.metrics_names:
        continue
    layer = model.keras_model.get_layer(name)
    model.keras_model.metrics_names.append(name)
    loss = (
        tf.reduce_mean(layer.output, keepdims=True)
        * config.LOSS_WEIGHTS.get(name, 1.))
    model.keras_model.add_metric(loss)


model.keras_model.compile(optimizer)

print("model.keras_model.input",model.keras_model.summary())

#Checkpoints
model_path = os.path.join(args.save_path, 'shapes')
if not os.path.exists(model_path):
    os.makedirs(model_path)

log = CSVLogger(os.path.join(model_path, 'shapes' + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True)
early_stop = EarlyStopping(monitor='loss', patience=3)


model.keras_model.fit(
    train_generator,
    epochs=args.num_epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=[log, checkpoint, early_stop],
    validation_data=val_generator,
    validation_steps=config.VALIDATION_STEPS,
    max_queue_size=100,
    workers=args.workers,
    use_multiprocessing=args.multiprocessing,
)