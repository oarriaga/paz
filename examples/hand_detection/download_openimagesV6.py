import fiftyone

fiftyone.zoo.load_zoo_dataset(
    'open-images-v6',
    split='train',
    label_types=['detections'],
    classes=['Human hand'])


fiftyone.zoo.load_zoo_dataset(
    'open-images-v6',
    split='validation',
    label_types=['detections'],
    classes=['Human hand'])


fiftyone.zoo.load_zoo_dataset(
    'open-images-v6',
    split='test',
    label_types=['detections'],
    classes=['Human hand'])
