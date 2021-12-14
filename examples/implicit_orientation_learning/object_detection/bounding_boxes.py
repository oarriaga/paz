import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from paz.backend.image import draw_rectangle


def draw_bounding_box(image, object_bounding_box):
    image = image.astype(float) / 255.
    thickness = 1
    image = draw_rectangle(image, [object_bounding_box[0] - thickness, object_bounding_box[1] - thickness], [object_bounding_box[2] + thickness, object_bounding_box[3] + thickness], (0, 1., 0), thickness)
    return image

def calc_bounding_box(image):
    object_mask = np.argwhere(np.sum(image, axis=-1) != 0)
    object_bounding_box = np.array([[object_mask[:, 1].min(), object_mask[:, 0].min(), object_mask[:, 1].max(), object_mask[:, 0].max()]])

    return object_bounding_box

def calc_bounding_boxes_existing_images(image_dir, image_size=128.):
    object_bounding_boxes = list()
    file_names = sorted(glob.glob(os.path.join(image_dir, "*")))

    for file_name in file_names:
        image = np.load(file_name)

        object_bounding_box = calc_bounding_box(image)/image_size
        object_bounding_boxes.append(object_bounding_box)

    object_bounding_boxes = np.asarray(object_bounding_boxes)
    return object_bounding_boxes


if __name__ == "__main__":
    object_bounding_boxes = calc_bounding_boxes_existing_images("/home/fabian/.keras/tless_obj05/pix2pose/normal_coloring/train/image_original")
    np.save("/home/fabian/.keras/tless_obj05/implicit_orientation/train/object_bounding_boxes.npy", object_bounding_boxes)

    #bounding_boxes = np.load("/home/fabian/Dokumente/object_bounding_boxes.npy")
    #image = np.load("/home/fabian/Dokumente/image_original_0009997.npy")


    #image = draw_bounding_box(image, np.squeeze(bounding_boxes[-3]).astype(int))
    #plt.imshow(image)
    #plt.show()