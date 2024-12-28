from paz.datasets import voc


def load(name, split):
    if name in ["VOC2007", "VOC2012"]:
        dataset = voc.load(name, split)
    else:
        raise ValueError
    return dataset
