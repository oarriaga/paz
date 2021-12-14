import os
import argparse
import glob
import tensorflow as tf
import numpy as np
import neptune
import random

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from paz.optimization.callbacks import LearningRateScheduler
from detection import AugmentDetection
from paz.models import SSD300
from paz.datasets import VOC
from paz.optimization import MultiBoxLoss
from paz.abstract import ProcessingSequence
from paz.optimization.callbacks import EvaluateMAP
from paz.pipelines import DetectSingleShot
from paz.processors import TRAIN, VAL

from callbacks import PlotImagesCallback, NeptuneLogger

description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bs', '--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('-is', '--image_size', default=128, type=int)
parser.add_argument('-et', '--evaluation_period', default=10, type=int,
                    help='evaluation frequency')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-g', '--gamma_decay', default=0.1, type=float,
                    help='Gamma decay for learning rate scheduler')
parser.add_argument('-e', '--num_epochs', default=240, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-iou', '--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-sp', '--save_path', default='trained_models/',
                    type=str, help='Path for writing model weights and logs')
#parser.add_argument('-dp', '--data_path', default='/media/fabian/Data/Masterarbeit/data/VOCdevkit',
#                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-dp', '--image_paths', nargs='+', default=['/home/fabian/.keras/tless_obj05/implicit_orientation'],
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-bi', '--background_images_directory', default="/home/fabian/.keras/backgrounds", type=str)
parser.add_argument('-ni', '--num_images_per_object', type=int, default=1000, help="Number of images per object to train on")
parser.add_argument('-se', '--scheduled_epochs', nargs='+', type=int,
                    default=[110, 152], help='Epoch learning rate reduction')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
parser.add_argument('-nc', '--neptune_config',
                    type=str, help='Path to config file where Neptune Token and project name is stored')
parser.add_argument('-nl', '--neptune_log_interval',
                    type=int, default=100, help='How long (in epochs) to wait for the next Neptune logging')
args = parser.parse_args()

optimizer = SGD(args.learning_rate, args.momentum)

background_image_paths = glob.glob(os.path.join(args.background_images_directory, '*.jpg'))

# Load data
data = [list(), list()]

for num_object, image_path in enumerate(args.image_paths):
    for i, mode in enumerate(["train", "test"]):

        file_names_images = sorted(glob.glob(os.path.join(image_path, mode, "image_original/*")))
        file_names_alpha = sorted(glob.glob(os.path.join(image_path, mode, "alpha_original/*")))
        object_bounding_boxes = np.load(os.path.join(image_path, mode, "object_bounding_boxes.npy"))

        # Add the object label
        object_bounding_boxes = np.concatenate((object_bounding_boxes, np.ones((len(object_bounding_boxes), 1, 1))*(num_object+1)), axis=-1)
        print(object_bounding_boxes.shape)

        if mode == "train":
            file_names_images = file_names_images[:args.num_images_per_object]
            file_names_alpha = file_names_alpha[:args.num_images_per_object]
            object_bounding_boxes = object_bounding_boxes[:args.num_images_per_object]
        else:
            file_names_images = file_names_images[:int(args.num_images_per_object/10)]
            file_names_alpha = file_names_alpha[:int(args.num_images_per_object/10)]
            object_bounding_boxes = object_bounding_boxes[:int(args.num_images_per_object/10)]

        for j, (file_name_image, file_name_alpha) in enumerate(zip(file_names_images, file_names_alpha)):
            data[i].append({"image": np.load(file_name_image), "boxes": object_bounding_boxes[j], "alpha_mask": np.load(file_name_alpha)})

    random.shuffle(data[i])


# instantiating model
num_classes = len(args.image_paths) + 1
model = SSD300(num_classes, base_weights='VGG', head_weights=None)
model.summary()

# Instantiating loss and metrics
loss = MultiBoxLoss()
metrics = {'boxes': [loss.localization,
                     loss.positive_classification,
                     loss.negative_classification]}
model.compile(optimizer, loss.compute_loss, metrics)

augmentator_train = AugmentDetection(model.prior_boxes, TRAIN, num_classes=num_classes, background_image_paths=background_image_paths)
sequencer_train = ProcessingSequence(augmentator_train, args.batch_size, data[0])

augmentator_test = AugmentDetection(model.prior_boxes, TRAIN, num_classes=num_classes, background_image_paths=background_image_paths)
sequencer_test = ProcessingSequence(augmentator_test, args.batch_size, data[1])

plotImagesCallback = PlotImagesCallback(model, data[1], neptune_logging=(args.neptune_config is not None), background_image_paths=background_image_paths)

callbacks = [plotImagesCallback]

# set up neptune run
if args.neptune_config is not None:
    neptune_config_file = open(args.neptune_config)
    neptune_config = neptune_config_file.read().split('\n')
    neptune_token = neptune_config[0]
    neptune_experiment_name = neptune_config[1]
    neptune_run_name = neptune_config[2]

    neptune.init(
       api_token=neptune_token,
       project_qualified_name=neptune_experiment_name
    )

    neptune.create_experiment(
       name=neptune_run_name,
       upload_stdout=False,
       params={'batch_size': args.batch_size, 'learning_rate': args.learning_rate}
    )

    neptuneCallback = NeptuneLogger(model, args.neptune_log_interval, args.save_path)
    callbacks.append(neptuneCallback)

# training
model.fit(
    sequencer_train,
    epochs=args.num_epochs,
    callbacks=callbacks,
    verbose=1,
    use_multiprocessing=args.multiprocessing,
    workers=args.workers)
