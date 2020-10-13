from cluttered_mnist import ClutteredMNIST
from STN import STN
from paz.backend.image import write_image


dataset_path = "mnist_cluttered_60x60_6distortions.npz"
batch_size = 256
num_epochs = 10
save_path = ''

data_manager = ClutteredMNIST(dataset_path)
train_data, val_data, test_data = data_manager.load()
x_train, y_train = train_data


model = STN()
model.compile(loss={'label': 'categorical_crossentropy'}, optimizer='adam')
model.summary()


def plot_predictions(samples):
    (lables, interpolations) = model.predict(samples)
    for arg, images in enumerate(zip(interpolations, samples)):
        interpolated, image = images
        interpolated = (interpolated * 255).astype('uint8')
        image = (image * 255).astype('uint8')
        write_image('images/interpolated_image_%03d.png' % arg, interpolated)
        write_image('images/original_image_%03d.png' % arg, image)


model.fit(x_train, y_train, batch_size, num_epochs, validation_data=val_data)
plot_predictions(test_data[0][:9])
