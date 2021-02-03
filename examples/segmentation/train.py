import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from shapes import Shapes
from model import UNET_VGG16
from tensorflow.keras.optimizers import Adam
from paz.abstract import ProcessingSequence, SequentialProcessor
from paz import processors as pr

num_classes = 3
input_shape = (128, 128, 3)
activation = 'sigmoid'
num_samples = 1000
iou_thresh = 0.3
max_num_shapes = 3
metrics = ['mean_squared_error']
loss = 'categorical_crossentropy'
H, W = image_shape = input_shape[:2]
batch_size = 5
epochs = 10
freeze = True

data_manager = Shapes(num_samples, image_shape, iou_thresh=iou_thresh,
                      max_num_shapes=max_num_shapes)
num_classes = data_manager.num_classes - 1
data = data_manager.load_data()
preprocess_image = SequentialProcessor()
preprocess_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
preprocess_image.add(pr.SubtractMeanImage(pr.BGR_IMAGENET_MEAN))
processor = SequentialProcessor()
processor.add(pr.UnpackDictionary(['image', 'masks']))
processor.add(pr.ControlMap(preprocess_image, [0], [0]))
processor.add(pr.SequenceWrapper({0: {'input_1': [H, W, 3]}},
                                 {1: {'masks': [H, W, num_classes]}}))

model = UNET_VGG16(num_classes, input_shape, 'imagenet', freeze, activation)
sequence = ProcessingSequence(processor, batch_size, data)
optimizer = Adam()
model.compile(optimizer, loss, metrics)
model.summary()
model.fit(sequence, batch_size=batch_size, epochs=epochs, callbacks=None)
