from paz.datasets import voc, shapes


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


def class_map(name):
    if name in ["VOC2007", "VOC2012"]:
        class_names = voc.get_class_names()
    elif name == "SHAPES":
        class_names = shapes.get_class_names()
    else:
        raise ValueError
    return build_arg_to_name(class_names)
