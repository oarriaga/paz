import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from paz.abstract import ProcessingSequence, SequentialProcessor
from loss import DiceLoss, JaccardLoss, FocalLoss
from paz import processors as pr

from model import UNET_VGG16
from cityscapes import CityScapes
from pipelines import PreprocessSegmentationIds
from pipelines import PostprocessSegmentationIds

num_classes = 3
input_shape = (128, 128, 3)
# softmax requires a background class and a background mask
activation = 'softmax'
# activation = 'sigmoid'
num_samples = 1000
iou_thresh = 0.3
max_num_shapes = 3
metrics = ['mean_squared_error']
loss = JaccardLoss()
# loss = [DiceLoss(), JaccardLoss(), FocalLoss()]
H, W = image_shape = input_shape[:2]
batch_size = 5
epochs = 10
freeze = True
stop_patience = 5
reduce_patience = 2
experiment_path = 'experiments/'

label_path = '/home/octavio/Downloads/dummy/gtFine/'
image_path = '/home/octavio/Downloads/dummy/RGB_images/leftImg8bit/'
data_manager = CityScapes(image_path, label_path, 'train')
data = data_manager.load_data()
num_classes = data_manager.num_classes
processor = PreprocessSegmentationIds(image_shape, num_classes)

# setting additional callbacks
callbacks = []
log_filename = os.path.join(experiment_path, 'optimization.log')
log = CSVLogger(log_filename)
stop = EarlyStopping('loss', patience=stop_patience)
save_filename = os.path.join(experiment_path, 'model.hdf5')
save = ModelCheckpoint(save_filename, 'loss', save_best_only=True)
plateau = ReduceLROnPlateau('loss', patience=reduce_patience)
callbacks.extend([log, stop, save, plateau])

model = UNET_VGG16(num_classes, input_shape, 'imagenet', freeze, activation)
sequence = ProcessingSequence(processor, batch_size, data)
optimizer = Adam()
model.compile(optimizer, loss, metrics)
model.summary()
model.fit(sequence, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

# colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# postprocess = PostprocessSegmentationIds(model, colors)
# for sample in data:
#     image = sample['image']
#     postprocess(image)
