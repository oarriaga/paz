import paz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset = "shapes"
dataset = "voc"

if dataset == "shapes":
    images, class_args, boxes, masks = paz.datasets.shapes.load(
        510, 510, 0.5, 5, 100
    )
    class_names = paz.datasets.shapes.get_class_names()
    cmap_mask = ListedColormap(["black", "red", "green", "blue"])
elif dataset == "voc":
    images, class_args, boxes, masks = paz.datasets.load(
        "VOC2007", "trainval", "segmentation"
    )
    class_names = paz.datasets.voc.get_class_names()
    cmap_mask = ListedColormap(
        [
            "black",
            "lightcoral",
            "red",
            "orangered",
            "chocolate",
            "gold",
            "olive",
            "greenyellow",
            "lightgreen",
            "navajowhite",
            "forestgreen",
            "lime",
            "aquamarine",
            "teal",
            "deepskyblue",
            "navy",
            "blueviolet",
            "violet",
            "magenta",
            "crimson",
            "pink",
            "white",
        ]
    )
    # cmap_mask = plt.cm.Paired
else:
    raise ValueError


def get_colormap_to_class():
    return {
        (0, 0, 0): 0,
        (128, 0, 0): 1,
        (0, 128, 0): 2,
        (128, 128, 0): 3,
        (0, 0, 128): 4,
        (128, 0, 128): 5,
        (0, 128, 128): 6,
        (128, 128, 128): 7,
        (64, 0, 0): 8,
        (192, 0, 0): 9,
        (64, 128, 0): 10,
        (192, 128, 0): 11,
        (64, 0, 128): 12,
        (192, 0, 128): 13,
        (64, 128, 128): 14,
        (192, 128, 128): 15,
        (0, 64, 0): 16,
        (128, 64, 0): 17,
        (0, 192, 0): 18,
        (128, 192, 0): 19,
        (0, 64, 128): 20,
        (224, 224, 192): 21,
    }


def color_map_to_class_arg(mask, colormap_to_class):
    class_masks = np.zeros(np.shape(mask)[:2])
    for color, class_arg in colormap_to_class.items():
        color = np.array(color).astype("uint8")
        is_class = np.all(mask == color, axis=-1)
        class_masks = np.where(is_class, class_arg, class_masks)
    return class_masks


# images_mosaic = paz.draw.mosaic(np.array(images), border=10)
# plt.imshow(images_mosaic)
# plt.show()

# masks_mosaic = paz.draw.mosaic(np.array(masks), border=10, background=4)
# cmap = ListedColormap(["black", "red", "green", "blue", "white"])
# plt.imshow(masks_mosaic, interpolation="nearest", cmap=cmap)
# plt.colorbar()
# plt.show()
# class_names = paz.datasets.shapes.get_class_names()
arg_to_name = paz.datasets.build_arg_to_name(class_names)
draw_box_args = ((255, 0, 0), 1, 1, 0.5)
colormap_to_class = get_colormap_to_class()
for arg in range(100):
    image = images[arg]
    mask = masks[arg]
    if dataset == "voc":
        image = np.array(paz.image.load(image))
        mask = np.array(paz.image.load(mask))
        mask = color_map_to_class_arg(mask, colormap_to_class)
    image_boxes = boxes[arg]
    image_class_args = class_args[arg]
    figure, axes = plt.subplots(1, 3)
    axes[0].imshow(image)
    names = [arg_to_name[class_arg] for class_arg in image_class_args]
    image_with_boxes = paz.draw.boxes(image, image_boxes, names, *draw_box_args)
    axes[1].imshow(image_with_boxes)
    axes[2].imshow(mask, interpolation="nearest", cmap=cmap_mask)
    plt.show()
