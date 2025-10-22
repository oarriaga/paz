from paz.datasets import (
    voc,
    shapes,
    fer,
    coco,
    fewsol,
    deepfish,
    omniglot,
    fsclvr,
)


def load(name, *args, **kwargs):
    if name in ["VOC2007", "VOC2012"]:
        dataset = voc.load(name, *args, **kwargs)
    elif name == "SHAPES":
        dataset = shapes.load(*args, **kwargs)
    else:
        raise ValueError
    return dataset


def build_arg_to_name(class_names):
    return dict(zip(range(len(class_names)), class_names))


def build_name_to_arg(class_names):
    return dict(zip(class_names, range(len(class_names))))


def labels(name):
    if name in ["VOC", "VOC2007", "VOC2012"]:
        class_names = voc.get_class_names()
    elif name == "SHAPES":
        class_names = shapes.get_class_names()
    elif name == "FER":
        class_names = fer.get_class_names()
    elif name == "COCO":
        class_names = coco.get_class_names()
    else:
        raise ValueError
    return class_names


def class_map(name):
    if name in ["VOC2007", "VOC2012"]:
        class_names = voc.get_class_names()
    elif name == "SHAPES":
        class_names = shapes.get_class_names()
    elif name == "FER":
        class_names = fer.get_class_names()
    else:
        raise ValueError
    return build_arg_to_name(class_names)


def get_intrinsics(name):
    if name.lower() == "fewsol":
        intrinsics = fewsol.get_intrinsics()
    elif name.lower() == "fsclvr":
        intrinsics = fsclvr.get_intrinsics()
    else:
        raise ValueError
    return intrinsics


def get_y_FOV(name):
    if name.lower() == "fewsol":
        y_FOV = fewsol.get_y_FOV()
    elif name.lower() == "fsclvr":
        y_FOV = fsclvr.get_y_FOV()
    else:
        raise ValueError
    return y_FOV
