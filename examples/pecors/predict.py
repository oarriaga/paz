import glob
import os

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from paz.abstract import GeneratingSequence
from pipelines import GeneratedImageGenerator, GeneratedVectorGenerator

def make_prediction(model_path, batch):
    # Threshold to distinguish between background and circle
    prediction_threshold = 0.1

    pecors = load_model(model_path)
    predictions = pecors.predict(batch[0]['input_image'])
    img_size = (predictions.shape[1], predictions.shape[2])

    for real_image, predicted_image in zip(batch[1]['circle_output'], predictions):
        arg_predicted_colors_filtered = np.argwhere(np.sum(predicted_image, axis=-1) > 3 * prediction_threshold)
        center_predicted_circle = np.array([np.mean(arg_predicted_colors_filtered[:, 1]), np.mean(arg_predicted_colors_filtered[:, 0])])
        print("Center: {}".format(center_predicted_circle))

        x_image = np.arange(0, img_size[0])
        y_image = np.arange(0, img_size[1])
        radius = 4
        predicted_circle_mask = (x_image[np.newaxis,:]-center_predicted_circle[0])**2 + (y_image[:,np.newaxis]-center_predicted_circle[1])**2 < radius**2

        predicted_colors_filtered = predicted_image[predicted_circle_mask]
        real_colors_filtered = real_image[np.sum(real_image, axis=-1) > 3 * prediction_threshold]

        real_vector = (np.mean(real_colors_filtered, axis=0)*2.0) - 1.0
        predicted_vector = (np.mean(predicted_colors_filtered, axis=0)*2.0) - 1.0

        plt.imshow(predicted_image)
        plt.show()

        plt.imshow(real_image)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.quiver(np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([real_vector[0], predicted_vector[0]]), np.array([real_vector[1], predicted_vector[1]]), np.array([real_vector[2], predicted_vector[2]]), length=1.0, normalize=True)
        plt.show()

        print("Prediction: {}".format(np.mean(predicted_colors_filtered, axis=0)))
        print("Real: {}".format(np.mean(real_colors_filtered, axis=0)))

    
def make_prediction_vectors(model_path, batch):
    pecors = load_model(model_path)
    predictions = pecors.predict(batch[0]['input_image'])

    print("Real rotation: {}".format(batch[1]['rotation_output']))
    print("Predicted rotation: {}".format(predictions[0]))

    print("Real translation: {}".format(batch[1]['translation_output']))
    print("Predicted translation: {}".format(predictions[1]))


if __name__ == '__main__':
    model_path = '/media/fabian/Data/Masterarbeit/data/models/tless03/pecors/vectors/pecors_model_epoch_9900.pkl'
    images_directory = '/media/fabian/Data/Masterarbeit/data/tless_obj03/vectors'
    background_images_directory = '/home/fabian/.keras/backgrounds'
    image_size = 128

    background_image_paths = glob.glob(os.path.join(background_images_directory, '*.jpg'))
    processor_test = GeneratedVectorGenerator(os.path.join(images_directory, "test"), background_image_paths,
                                             image_size=image_size, num_occlusions=0)
    sequence_test = GeneratingSequence(processor_test, 20, 1)
    batch = sequence_test.__getitem__(0)

    make_prediction_vectors(model_path, batch)