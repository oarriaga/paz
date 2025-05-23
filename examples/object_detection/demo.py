import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
import jax.numpy as jp
import numpy as np
import paz
import cv2


image = paz.image.load("photo_2.jpg")
# pipeline = paz.applications.SSD300VOC()
# image, boxes2D = pipeline(image)
# paz.image.show(image)

boxes = jp.array(
    [[567, 439, 976, 844], [224, 327, 866, 717], [611, 267, 907, 792]]
)
labels = jp.array([1, 6, 14])
scores = jp.array([0.99148387, 0.9997619, 0.7915614])


def draw_boxes2D(image, boxes, class_args, scores, names, colors, thickness):
    font_scale = 0.7
    font = cv2.FONT_HERSHEY_DUPLEX
    image = np.ascontiguousarray(np.array(image, dtype=image.dtype))
    for box, class_arg, score in zip(boxes, class_args, scores):
        color = colors[class_arg]
        x_min, y_min, x_max, y_max = box = box.tolist()
        image = paz.draw.box(image, box, colors[class_arg], thickness)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        label = f"{names[class_arg]} {score * 100:.0f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        offset = round(thickness / 2)
        cv2.rectangle(
            image,
            (x_min - offset, y_min - text_height - baseline - thickness),
            (x_min + text_width, y_min),
            color,
            -1,
        )

        cv2.putText(
            image,
            label,
            (x_min, y_min - baseline),
            font,
            font_scale,
            (255, 255, 255),
        )

    return image, (boxes, class_args, scores)


names = paz.datasets.labels("VOC")
colors = paz.draw.lincolor(len(names))
image_with_boxes, boxes2D = draw_boxes2D(
    image, boxes, labels, scores, names, colors, 3
)
paz.image.show(image_with_boxes)
