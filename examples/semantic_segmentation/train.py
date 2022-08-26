import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from paz.abstract import ProcessingSequence
from paz.optimization import DiceLoss, JaccardLoss, FocalLoss
from paz.models import UNET_VGG16
# from paz import processors as pr
from paz.datasets import Shapes
from pipelines import PreprocessSegmentation
from pipelines import PostprocessSegmentation

num_classes = 3
input_shape = (128, 128, 3)
# softmax requires a background class and a background mask
activation = 'softmax'
# activation = 'sigmoid'
num_samples = 1000
iou_thresh = 0.3
max_num_shapes = 3
metrics = ['mean_squared_error']
# loss = JaccardLoss()
loss = [DiceLoss(), JaccardLoss(), FocalLoss()]
H, W = image_shape = input_shape[:2]
batch_size = 5
epochs = 10
freeze = True
stop_patience = 5
reduce_patience = 2
experiment_path = 'experiments/'

data_manager = Shapes(num_samples, image_shape, iou_thresh=iou_thresh,
                      max_num_shapes=max_num_shapes)
num_classes = data_manager.num_classes
data = data_manager.load_data()
processor = PreprocessSegmentation(image_shape, num_classes)

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

colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
postprocess = PostprocessSegmentation(model, colors)
for sample in data:
    image = sample['image']
    postprocess(image)
