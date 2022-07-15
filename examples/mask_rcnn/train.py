import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint

from mask_rcnn.config import Config
from mask_rcnn.pipeline import DetectionPipeline#, DataSequencer
from paz.models.detection.utils import create_prior_boxes
#from ycb import YCBVideo
from mask_rcnn.loss import Loss
from mask_rcnn.network import create_network_head
from mask_rcnn.utils import data_generator
from paz.datasets.shapes import Shapes
from mask_rcnn.model import MaskRCNN, get_imagenet_weights


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    """
    NAME = 'shapes'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    NUM_CLASSES = 1 + 3  # background + 3 shapes
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5


description = 'Training script for Mask RCNN model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bs', '--batch_size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('-dp', '--data_path', required=False,
                    type=str, help='Directory for loading data')
parser.add_argument('-sp', '--save_path', required=False,
                    metavar='/path/to/save',
                    help="Path to save model weights and logs")
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-st', '--steps_per_epoch', default=1000, type=int,
                    help='steps per epoch for training')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-e', '--num_epochs', default=120, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-et', '--evaluation_period', default=1, type=int,
                    help='evaluation frequency')
parser.add_argument('--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-l', '--layers', default='all', type=str,
                    help='Select which layers to train')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
args = parser.parse_args()
print('Path to save model: ', args.save_path)
print('Data path: ', args.data_path)

optimizer = SGD(args.learning_rate, args.momentum)
config = ShapesConfig()

# Training data
dataset_train = Shapes(500, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))
dataset_train.load_data()
#dataset_train.prepare() #prepare function added extra, need to check the impact

# Validation data
dataset_val = Shapes(50, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))
dataset_val.load_data()
#dataset_val.prepare()  #prepare function check added extra, need to check the  impact

train_generator = data_generator(dataset_train, config, shuffle=True,
                                 augmentation=None,
                                 batch_size=config.BATCH_SIZE)
val_generator = data_generator(dataset_val, config,
                               shuffle=True, batch_size=config.BATCH_SIZE)

# instantiating model
num_classes = config.NUM_CLASSES
model = MaskRCNN(config=config, model_dir=args.data_path, train_bn=config.TRAIN_BN,
                 image_shape=config.IMAGE_SHAPE, backbone=config.BACKBONE,
                 top_down_pyramid_size=config.TOP_DOWN_PYRAMID_SIZE)
model.keras_model = create_network_head(model, config)
model.keras_model.load_weights(get_imagenet_weights(), by_name=True)

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

# setting detection pipeline
detectors = []
prior_boxes = create_prior_boxes('YCBVideo')
for split in ['TRAIN', 'VAL']:
    detector = DetectionPipeline(config, prior_boxes, num_classes=num_classes)
    detectors.append(detector)

# setting sequencers
# sequencers = []
# for data, detector in zip(datasets, detectors):
#     sequencer = DataSequencer(detector, args.batch_size, data)
#     sequencers.append(sequencer)

model_path = os.path.join(args.save_path, 'shapes')
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(os.path.join(model_path, 'shapes' + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True)
early_stop = EarlyStopping(monitor='loss', patience=3)

model.set_trainable(layer_regex=layers)

# keras_model._losses = []
# keras_model._per_input_losses = {}
# loss_names = ['rpn_class_loss',  'rpn_bbox_loss',
#               'mrcnn_class_loss', 'mrcnn_bbox_loss', 'mrcnn_mask_loss']
# added_loss_name = []

# for name in loss_names:
#     layer = model.keras_model.get_layer(name)
#     if layer.output.name in added_loss_name:
#         continue
#     loss = (
#         tf.reduce_mean(input_tensor=layer.output, keepdims=True)
#         * config.LOSS_WEIGHTS.get(name, 1.))
#     model.keras_model.add_loss(loss)
#     added_loss_name.append(layer.output.name)

# l2 Regularization

reg_losses = [
    l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
    for w in model.keras_model.trainable_weights
    if 'gamma' not in w.name and 'beta' not in w.name]
model.keras_model.add_loss(lambda: tf.add_n(reg_losses))
model.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(model.keras_model.outputs))

loss = Loss(config)
losses = {'rpn_class_logits': loss.rpn_class_loss_graph,
          'rpn_bbox': loss.rpn_bbox_loss_graph,
          'mrcnn_class': loss.mrcnn_class_loss_graph,
          'mrcnn_bbox': loss.mrcnn_bbox_loss_graph,
          'mrcnn_mask': loss.mrcnn_mask_loss_graph}

metrics = {'rpn_class_logits': loss.rpn_class_metrics,
           'rpn_bbox': loss.rpn_bbox_metrics,
           'mrcnn_class': loss.mrcnn_class_metrics,
           'mrcnn_bbox': loss.mrcnn_bbox_metrics,
           'mrcnn_mask': loss.mrcnn_mask_metrics}

model.keras_model.compile(optimizer, losses, metrics=metrics)

# for name in loss_names:
#     if name in model.keras_model.metrics_names:
#         continue
#     layer = model.keras_model.get_layer(name)
#     model.keras_model.metrics_names.append(name)
#     loss = (
#         tf.math.reduce_mean(input_tensor=layer.output, keepdims=True)
#         * config.LOSS_WEIGHTS.get(name, 1.))
#     model.keras_model.add_metric(loss, name=name, aggregation='mean')


model.keras_model.fit_generator(
    dataset_train,
    epochs=args.epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=[log, checkpoint, early_stop],
    validation_data=dataset_val,
    validation_steps=config.VALIDATION_STEPS,
    max_queue_size=100,
    workers=args.workers,
    use_multiprocessing=args.multiprocessing,
)
