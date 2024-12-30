from paz.datasets import voc, shapes


def load(name, **kwargs):
    if name in ["VOC2007", "VOC2012"]:
        dataset = voc.load(name, **kwargs)
    elif name == "SHAPES":
        dataset = shapes.load(**kwargs)
    else:
        raise ValueError
    return dataset


def build_arg_to_name(class_names):
    return dict(zip(range(len(class_names)), class_names))


def build_name_to_arg(class_names):
    return dict(zip(class_names, range(len(class_names))))
