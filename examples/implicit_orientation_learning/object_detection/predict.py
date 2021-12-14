import matplotlib.pyplot as plt
import numpy as np
import glob
import os

from scenes import SingleView
from bounding_boxes import draw_bounding_box
from paz.pipelines.detection import DetectSingleShot
from paz.processors import SequentialProcessor
from paz.pipelines import RandomizeRenderedImage
import paz.processors as pr
from paz.models import SSD300


def calculate_translation(bounding_box, camera_intrinsics_matrix, object_translation, matrices):
    camera_to_world = np.reshape(matrices[1], (4, 4))
    print(camera_to_world)

    predicted_z = object_translation[-1] - camera_to_world[2, 3]

    predicted_x = -predicted_z * (bounding_box.center[0] - camera_intrinsics_matrix[0, 2]) / camera_intrinsics_matrix[0, 0]
    predicted_y = predicted_z * (bounding_box.center[1] - camera_intrinsics_matrix[1, 2]) / camera_intrinsics_matrix[1, 1]

    print("Predicted translation: {}".format(np.array([predicted_x, predicted_y, predicted_z])))
    print("Real translation: {}".format(object_translation - np.array([0, 0, camera_to_world[2, 3]])))

    return np.array([predicted_x, predicted_y, predicted_z])


if __name__ == "__main__":
    filepath_obj = "/home/fabian/.keras/datasets/tless_obj/tless14.obj"
    model_weights = "/media/fabian/Data/Masterarbeit/data/models/ssd_implicit_orientation/ssd_300_detection_20_weights.h5"
    background_images_path = "/home/fabian/.keras/backgrounds"
    img_size = (128., 128.)

    focal_length = 155
    image_center = (img_size[1] / 2.0, img_size[0] / 2.0)

    camera_intrinsics_matrix = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    renderer = SingleView(filepath=filepath_obj, light=[0.5, 30], distance=[0.5, 0.8])

    model = SSD300(4, base_weights='VGG', head_weights=None)
    model.summary()
    model.load_weights(model_weights)

    background_image_paths = glob.glob(os.path.join(background_images_path, '*.jpg'))

    image, alpha_mask, object_bounding_box, object_translation, matrices = renderer.render()

    pipeline = SequentialProcessor([pr.CastImage(np.single),
                                    DetectSingleShot(model, ["Background", "TLESS 06", "TLESS 14", "TLESS 27"], 0.6, 0.45),
                                    pr.UnpackDictionary(['image', 'boxes2D']),
                                    pr.ControlMap(pr.CastImage(np.int), [0], [0])])
    augment_background = RandomizeRenderedImage(image_paths=background_image_paths, num_occlusions=0)

    image = augment_background(image, alpha_mask)
    prediction = pipeline(image)

    predicted_translation = calculate_translation(prediction[1][0], camera_intrinsics_matrix, object_translation, matrices)

    print("Predicted box: {}".format(prediction[1]))

    plt.imshow(prediction[0])
    plt.show()