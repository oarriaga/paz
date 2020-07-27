import os
import json
import argparse

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from paz.abstract import ProcessingSequence

from facial_keypoints import FacialKeypoints
from pipelines import AugmentKeypoints
from model import GaussianMixtureModel

description = 'Training script for learning 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=8, type=int,
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
parser.add_argument('-nk', '--num_keypoints', default=15, type=int,
                    help='Number of keypoints')
parser.add_argument('-ds', '--delta_scales', default=0.2, type=float,
                    help='Delta scales')
parser.add_argument('-is', '--image_size', default=96, type=int,
                    help='Image size')
parser.add_argument('-r', '--rotation_range', default=30, type=float,
                    help='Rotation range')
parser.add_argument('-vs', '--validation_split', default=0.2, type=float,
                    help='Fraction of the training set used for validation')
parser.add_argument('-s', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
args = parser.parse_args()


# loading training dataset
data_manager = FacialKeypoints('dataset/', 'train')
data = data_manager.load_data()

# split training data-set into train and validation
num_train_samples = int(len(data) * (1 - args.validation_split))
datasets = {'train': data[:num_train_samples],
            'validation': data[num_train_samples:]}

# instantiate keypoint augmentations
delta_scales = [args.delta_scales, args.delta_scales]
processor = {}
for phase in ['train', 'validation']:
    processor[phase] = AugmentKeypoints(
        phase, args.rotation_range, delta_scales, True, args.num_keypoints)

# creating sequencers
sequence = {}
for phase in ['train', 'validation']:
    pipeline, data = processor[phase], datasets[phase]
    sequence[phase] = ProcessingSequence(pipeline, args.batch_size, data, True)

# instantiate model
batch_shape = (args.batch_size, args.image_size, args.image_size, 1)
model = GaussianMixtureModel(batch_shape, args.num_keypoints, args.filters)
model.summary()


# creating loss function for gaussian mixture model
def negative_log_likelihood(y_true, predicted_distributions):
    log_likelihood = predicted_distributions.log_prob(y_true)
    return - log_likelihood


# setting optimizer and compiling model
optimizer = Adam(args.learning_rate, amsgrad=True)
model.compile(optimizer, loss=negative_log_likelihood)

# making directory for saving model weights and logs
model_name = ['FaceKP', model.name, str(args.filters), str(args.num_keypoints)]
model_name = '_'.join(model_name)
save_path = os.path.join(args.save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# setting callbacks
log = CSVLogger(os.path.join(save_path, '%s.log' % model_name))
stop = EarlyStopping(patience=args.stop_patience, verbose=1)
plateau = ReduceLROnPlateau(patience=args.plateau_patience, verbose=1)
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
save = ModelCheckpoint(model_path, verbose=1, save_best_only=True,
                       save_weights_only=True)

# saving hyper-parameters and model summary as text files
with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(save_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

# model optimization
model.fit_generator(
    sequence['train'],
    epochs=args.max_num_epochs,
    callbacks=[stop, log, save, plateau],
    validation_data=sequence['validation'],
    verbose=1)
