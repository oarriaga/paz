class Loader(object):
    """Abstract class for loading a dataset.

    # Arguments
        path: String. Path to data.
        split: String. Dataset split e.g. traing, val, test.
        class_names: List of strings. Label names of the classes.
        name: String. Dataset name.

    # Properties
        name: Str.
        path: Str.
        split: Str or Flag.
        class_names: List of strings.
        num_classes: Int.

    # Methods
        load_data()
    """
    def __init__(self, path, split, class_names, name):
        self.path = path
        self.split = split
        self.class_names = class_names
        self.name = name

    def load_data(self):
        """Abstract method for loading dataset.

        # Returns
            dictionary containing absolute image paths as keys, and
            ground truth vectors as values.
        """
        raise NotImplementedError()

    # Name of the dataset (VOC2007, COCO, OpenImagesV4, etc)
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    #  Path to the dataset, ideally loaded from a configuration file.
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    # Kind of split to use, either train, validation, test, or trainval.
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        self._split = split

    # List of class names to train/test.
    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        # assert type(class_names) == list
        self._class_names = class_names

    @property
    def num_classes(self):
        if isinstance(self.class_names, list):
            return len(self.class_names)
        else:
            raise ValueError('class names are not a list')
