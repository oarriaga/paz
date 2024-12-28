import tensorflow as tf
import paz


images, masks, boxes, class_args = paz.datasets.load("VOC2007", "trainval")

boxes = paz.boxes.pad_data(boxes, 32)
class_args = paz.classes.pad_data(class_args, 32)
data = tf.data.Dataset.from_tensor_slices((images, masks, boxes, class_args))

