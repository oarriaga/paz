import os
from xml.etree import ElementTree
from functools import partial

from keras.utils import get_file

import paz


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


def validate_inputs(name, split, task):
    assert name in ["VOC2007", "VOC2012"]
    assert split in ["trainval", "test"]
    assert task in ["detection", "segmentation"]


def parse_line(path, name, line):
    return os.path.join(path, name, 'Annotations', line.strip() + ".xml")


def build_label_path(path, name, split, task):
    task = "Segmentation" if task == "segmentation" else "Main"
    return os.path.join(path, name, f"ImageSets/{task}/{split}.txt")


def get_label_paths(path, name, split, task):
    split_filename = build_label_path(path, name, split, task)
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


def strip_extension(file_path):
    base_name, _ = os.path.splitext(file_path)
    return base_name


def load(name, split="trainval", task="detection"):
    """Loads VOC2007 or VOC2012 with the trainval or test split.

    # Arguments
        name: String. Either `VOC2007` or `VOC2012`.
        split: String. Either `trainval` or `test`.
        task: String. Either `detection` or `segmentation`.

    # Returns
        Lists of image paths, boxes in xyxy format, and classes arguments.
    """
    validate_inputs(name, split, task)
    path = download(name, split)
    image_root = os.path.join(path, name, "JPEGImages")
    masks_root = os.path.join(path, name, "SegmentationClass")
    class_names = get_class_names()
    parse = partial(parse_XML, paz.datasets.build_name_to_arg(class_names))
    image_paths, masks_paths, boxes, class_args = [], [], [], []
    for label_path in get_label_paths(path, name, split, task):
        image_name, image_boxes, image_class_args = parse(label_path)
        if len(image_boxes) != 0:
            image_name = strip_extension(image_name)
            image_paths.append(os.path.join(image_root, image_name + ".jpg"))
            masks_paths.append(os.path.join(masks_root, image_name + ".png"))
            boxes.append(image_boxes)
            class_args.append(image_class_args)
        else:
            print(f"Image {image_name} had not boxes.")
    return image_paths, masks_paths, boxes, class_args


def colormap_to_class():
    return {
        (0, 0, 0, 255): 0,         # Black
        (128, 0, 0, 255): 1,       # Maroon
        (0, 128, 0, 255): 2,       # Green
        (128, 128, 0, 255): 3,     # Olive
        (0, 0, 128, 255): 4,       # Navy
        (128, 0, 128, 255): 5,     # Purple
        (0, 128, 128, 255): 6,     # Teal
        (128, 128, 128, 255): 7,   # Fractal
        (64, 0, 0, 255): 8,        # sRGBA(64,0,0,1)
        (192, 0, 0, 255): 9,       # sRGBA(192,0,0,1)
        (64, 128, 0, 255): 10,     # sRGBA(64,128,0,1)
        (192, 128, 0, 255): 11,    # sRGBA(192,128,0,1)
        (64, 0, 128, 255): 12,     # sRGBA(64,0,128,1)
        (192, 0, 128, 255): 13,    # sRGBA(192,0,128,1)
        (64, 128, 128, 255): 14,   # sRGBA(64,128,128,1)
        (192, 128, 128, 255): 15,  # sRGBA(192,128,128,1)
        (0, 64, 0, 255): 16,       # sRGBA(0,64,0,1)
        (128, 64, 0, 255): 17,     # sRGBA(128,64,0,1)
        (0, 192, 0, 255): 18,      # sRGBA(0,192,0,1)
        (128, 192, 0, 255): 19,    # sRGBA(128,192,0,1)
        (0, 64, 128, 255): 20,     # sRGBA(0,64,128,1)
        (224, 224, 192, 255): 21   # sRGBA(128,64,128,1)
    }
