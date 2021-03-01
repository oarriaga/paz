import argparse
import math
import os

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

from frustum_loader import frustum_data_loader, make_train_iterator
from paz.models.detection.frustum_pointnet import FrustumPointNetModel

description = 'Training script for learning 3D Object Detection'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-dr', '--decay_rate', default=0.5, type=float,
                    help='Decay rate for scheduling the learning rate')
parser.add_argument('-e', '--max_num_epochs', default=10000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-sp', '--stop_patience', default=45, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('-mn', '--model_name',
                    default='frustumpointnet_carpedcyc',
                    type=str, help='Model name based on object of interest')
parser.add_argument('-s', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-td', '--train_tfrec_path',
                    default='./frustum_dataset/',
                    type=str, help='Path for datasets generated')
parser.add_argument('-vd', '--val_tfrec_path',
                    default='./frustum_dataset/',
                    type=str, help='Path for datasets generated')
args = parser.parse_args()


def step_decay(epoch):
    initial_lrate = args.learning_rate
    drop = args.decay_rate
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


# loading data
data_loader = frustum_data_loader()

parsed_train_dataset = data_loader.load_data(args.train_tfrec_path,
                                             operation='train')
train_iterator = make_train_iterator(parsed_train_dataset)

parsed_validation_dataset = data_loader.load_data(args.val_tfrec_path,
                                                  operation='validation')
val_iterator = make_train_iterator(parsed_validation_dataset)

model_name = args.model_name

# setting callbacks
log = CSVLogger(os.path.join(args.save_path, '%s.log' % model_name))

stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0,
                     mode='auto', baseline=None, restore_best_weights=False)

lrate = LearningRateScheduler(step_decay)

model_path = os.path.join(args.save_path, '%s_weights.hdf5' % model_name)
save = ModelCheckpoint(model_path, verbose=1, save_best_only=True,
                       save_weights_only=True)

# model importing
model, _ = FrustumPointNetModel()
model.compile(optimizer=Adam(args.learning_rate),
              loss={'fp_loss': lambda y_true, y_pred: y_pred})

# model optimization
model.fit_generator(train_iterator, epochs=args.max_num_epochs,
                    callbacks=[stop, log, save, lrate],
                    validation_data=val_iterator,
                    verbose=1)
