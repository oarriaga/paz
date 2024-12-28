import os
from xml.etree import ElementTree
from functools import partial

from keras.utils import get_file


def get_class_names():
    return [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]


def build_class_map():
    class_names = get_class_names()
    return dict(zip(range(len(class_names)), class_names))


def build_class_map_inverse():
    class_names = get_class_names()
    return dict(zip(class_names, range(len(class_names))))


def parse_box(box):
    # matlab start indices at 1 thus we need to subtract 1
    x_min = int(box.find('xmin').text) - 1
    y_min = int(box.find('ymin').text) - 1
    x_max = int(box.find('xmax').text) - 1
    y_max = int(box.find('ymax').text) - 1
    return [x_min, y_min, x_max, y_max]


def parse_XML(name_to_arg, XML_filename):
    tree = ElementTree.parse(XML_filename)
    image_name = tree.find("filename").text
    boxes, class_args = [], []
    for detection in tree.findall("object"):
        class_arg = name_to_arg[detection.find("name").text]
        difficult = int(detection.find('difficult').text)
        box = parse_box(detection.find("bndbox"))
        if not difficult:
            boxes.append(box)
            class_args.append(class_arg)
    return image_name, boxes, class_args


def validate_inputs(name, split):
    assert name in ["VOC2007", "VOC2012"]
    assert split in ["trainval", "test"]


def parse_line(path, name, line):
    return os.path.join(path, name, 'Annotations', line.strip() + ".xml")


def get_label_paths(path, name, split):
    split_filename = os.path.join(path, name, f"ImageSets/Main/{split}.txt")
    return [parse_line(path, name, line) for line in open(split_filename)]


def get_URL(name, split):
    base_URL = "http://host.robots.ox.ac.uk/pascal/VOC/"
    if ((name == "VOC2007") and (split == "trainval")):
        URL = base_URL + "voc2007/VOCtrainval_06-Nov-2007.tar"
    elif ((name == "VOC2012") and (split == "trainval")):
        URL = base_URL + "voc2012/VOCtrainval_11-May-2012.tar"
    elif ((name == "VOC2007") and (split == "test")):
        URL = base_URL + "voc2007/VOCtest_06-Nov-2007.tar"
    return URL


def download(name, split):
    origin = get_URL(name, split)
    filepath = get_file(origin=origin, extract=True)
    filepath = os.path.join(os.path.dirname(filepath), "VOCdevkit")
    return filepath


def load(name, split):
    """Loads VOC2007 or VOC2012 with the trainval or test split.

    # Arguments
        name: String. Either `VOC2007` or `VOC2012`.
        name: String. Either `trainval` or `test`.

    # Returns
        Lists of image paths, boxes in xyxy format, and classes arguments.
    """
    validate_inputs(name, split)
    path = download(name, split)
    label_paths = get_label_paths(path, name, split)
    image_root = os.path.join(path, name, "JPEGImages")
    parse = partial(parse_XML, build_class_map_inverse())
    image_paths, boxes, class_args = [], [], []
    for label_path in label_paths:
        image_name, image_boxes, image_class_args = parse(label_path)
        if len(image_boxes) != 0:
            image_paths.append(os.path.join(image_root, image_name))
            boxes.append(image_boxes)
            class_args.append(image_class_args)
        else:
            print(f"Image {image_name} had not boxes.")
    return image_paths, boxes, class_args
