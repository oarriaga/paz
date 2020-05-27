from paz.datasets import VOC
import tensorflow as tf

data_manager = VOC('VOCdevkit/')
data = data_manager.load_data()

boxes, images = [], []
for sample in data:
    boxes.append(sample['boxes'])
    images.append(sample['image'])

boxes = tf.ragged.constant(boxes)
dataset = tf.data.Dataset.from_tensor_slices((images, boxes))
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocessing(filepath, boxes):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, 3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (300, 300))
    return image, boxes


def prepare_for_training(dataset, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


# AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = dataset.map(preprocessing, num_parallel_calls=None)
dataset = prepare_for_training(dataset)
dataset = iter(dataset)

