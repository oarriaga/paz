import os
import json
import argparse

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from paz.pipelines import KeypointAugmentation
from paz.pipelines import KeypointInference
from paz.core.sequencer import GeneratingSequencer
from paz.optimization.callbacks import DrawInferences

from models import GaussianMixture

description = 'Training script for learning 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=64, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-sp', '--stop_patience', default=7, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('-pp', '--plateau_patience', default=3, type=int,
                    help='Number of epochs before reducing learning rate')
parser.add_argument('-e', '--max_num_epochs', default=10000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-se', '--steps_per_epoch', default=1000, type=int,
                    help='Steps per epoch')
parser.add_argument('-o', '--num_occlusions', default=2, type=int,
                    help='Number of occlusions')
parser.add_argument('-s', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
args = parser.parse_args()


# creating loss function for gaussian mixture model
def negative_log_likelihood(y_true, predicted_distributions):
    log_likelihood = predicted_distributions.log_prob(y_true)
    return - log_likelihood


with_partition = True
batch_shape = (args.batch_size, args.image_size, args.image_size, 3)
loss = negative_log_likelihood
model = GaussianMixture(batch_shape, num_keypoints)


# setting optimizer and compiling model
optimizer = Adam(args.learning_rate, amsgrad=True)
model.compile(optimizer, loss=loss)
model.summary()


# setting scene
processor = KeypointAugmentation(renderer, projector, keypoints, 'train',
                                 image_paths, args.image_size, with_partition,
                                 args.num_occlusions)


# creating sequencer
sequencer = GeneratingSequencer(processor, args.batch_size)


# making directory for saving model weights and logs
model_name = '_'.join([model.name, str(len(keypoints)), args.class_name])
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
inferencer = KeypointInference(model, num_keypoints)
draw = DrawInferences(save_path, images, inferencer)


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
