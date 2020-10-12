import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizer import SGD
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from paz.datasets.ycb_video import YCBVideo

from mask_rcnn.model import MaskRCNN
from mask_rcnn.config import Config
from mask_rcnn.utils import data_generator


class YCBVideoConfig(Config):
    """Configuration for training on YCB-Video dataset.
    Derives from the base Config class and overrides values specific
    to the YCB-Video dataset.
    """
    NAME = "ycb"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 21


description = 'Training script for Mask RCNN model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-dp', '--data_path', required=False,
                    type=str, help='Directory for loading data')
parser.add_argument('-sp', '--save_path', required=False,
                    metavar='/path/to/save',
                    help="Path to save model weights and logs")
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-st', '--steps_per_epoch', default=1000, type=int,
                    help='Batch size for training')
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

data_splits = ['train', 'val', 'test']

data_managers, datasets, evaluation_data_managers = [], [], []
for data_split in data_splits:
    data_manager = YCBVideo(args.data_path, data_split)
    data_managers.append(data_manager)
    datasets.append(data_manager.load_data())
    if data_split == 'test':
        eval_data_manager = YCBVideo(args.data_path, data_split)
        evaluation_data_managers.append(eval_data_manager)

# instantiating model
num_classes = data_managers[0].num_classes
config = YCBVideoConfig()
model = MaskRCNN(config=config, model_dir=args.data_path)
model.keras_model.load_weights(model.get_imagenet_weights())
model.summary()

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
if args.layers in layer_regex.keys():
    layers = layer_regex[args.layers]


train_generator = data_generator(data_managers[0], config, shuffle=True,
                                 augmentation=None,
                                 batch_size=config.BATCH_SIZE)
val_generator = data_generator(evaluation_data_managers[0], config,
                               shuffle=True,
                               batch_size=config.BATCH_SIZE)

model_path = os.path.join(args.save_path, 'ycb')
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(os.path.join(model_path, 'ycb' + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True)
early_stop = EarlyStopping(monitor='loss', patience=3)

model.set_trainable(layers)

loss_names = ['rpn_class_loss',  'rpn_bbox_loss',
              'mrcnn_class_loss', 'mrcnn_bbox_loss', 'mrcnn_mask_loss']
added_loss_name = []

for name in loss_names:
    layer = model.keras_model.get_layer(name)
    if layer.output.name in added_loss_name:
        continue
    loss = (
        tf.reduce_mean(input_tensor=layer.output, keepdims=True)
        * config.LOSS_WEIGHTS.get(name, 1.))
    model.keras_model.add_loss(loss)
    added_loss_name.append(layer.output.name)

# L2 Regularization
reg_losses = [
    L2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
    for w in model.keras_model.trainable_weights
    if 'gamma' not in w.name and 'beta' not in w.name]
model.keras_model.add_loss(lambda: tf.add_n(reg_losses))
model.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(model.keras_model.outputs))

for name in loss_names:
    if name in model.keras_model.metrics_names:
        continue
    layer = model.keras_model.get_layer(name)
    model.keras_model.metrics_names.append(name)
    loss = (
        tf.math.reduce_mean(input_tensor=layer.output, keepdims=True)
        * config.LOSS_WEIGHTS.get(name, 1.))
    model.keras_model.add_metric(loss, name=name, aggregation='mean')


model.keras_model.fit_generator(
    train_generator,
    epochs=args.epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=[log, checkpoint, early_stop],
    validation_data=val_generator,
    validation_steps=config.VALIDATION_STEPS,
    max_queue_size=100,
    workers=args.workers,
    use_multiprocessing=args.multiprocessing,
)
