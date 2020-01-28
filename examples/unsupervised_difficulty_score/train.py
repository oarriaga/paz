from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D

from paz.core import ProcessingSequencer


from callbacks import Evaluator
from mnist import MNIST, ImageAugmentation

size, num_classes, batch_size, epochs, save_path = 28, 10, 32, 10, 'data'
splits, evaluations_filename = ['train', 'test'], 'evaluations.hdf5'

sequencers = {}
for split in splits:
    processor = ImageAugmentation(size, num_classes)
    sequencer = ProcessingSequencer(processor, batch_size, MNIST(split).load())
    sequencers[split] = sequencer
input_shape = sequencers['train'].processor.input_shapes[0]


inputs = Input(input_shape, name='image')
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax', name='label')(x)
model = Model(inputs, outputs, name='CNN')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

evaluators = [keras.losses.MeanSquaredError('none')]
evaluate = Evaluator(sequencers['test'], 'label', evaluators,
                     epochs, evaluations_filename)

model.fit_generator(
    sequencers['train'],
    steps_per_epoch=100,
    epochs=epochs,
    callbacks=[evaluate],
    verbose=1,
    validation_data=sequencers['test'])
