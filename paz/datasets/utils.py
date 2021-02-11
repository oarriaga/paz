def get_class_names(dataset_name='VOC2007'):
    """Gets label names for the classes of the supported datasets.

    # Arguments
        dataset_name: String. Dataset name. Valid dataset names are:
            VOC2007, VOC2012, COCO and YCBVideo.

    # Returns
       List of strings containing the class names for the dataset given.

    # Raises
        ValueError: in case of invalid dataset name
    """

    if dataset_name in ['VOC2007', 'VOC2012', 'VOC']:

        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    elif dataset_name == 'COCO':
        class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle',
                       'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign',
                       'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                       'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard',
                       'tennis racket', 'bottle', 'wine glass',
                       'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                       'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet',
                       'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                       'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    elif dataset_name == 'YCBVideo':
        class_names = ['background', '037_scissors', '008_pudding_box',
                       '024_bowl', '005_tomato_soup_can', '007_tuna_fish_can',
                       '010_potted_meat_can', '061_foam_brick', '011_banana',
                       '035_power_drill', '004_sugar_box', '019_pitcher_base',
                       '006_mustard_bottle', '036_wood_block',
                       '009_gelatin_box', '051_large_clamp',
                       '040_large_marker', '003_cracker_box',
                       '025_mug', '052_extra_large_clamp',
                       '021_bleach_cleanser', '002_master_chef_can']

    elif dataset_name == 'FAT':
        class_names = ['background', '037_scissors', '008_pudding_box',
                       '024_bowl', '005_tomato_soup_can', '007_tuna_fish_can',
                       '010_potted_meat_can', '061_foam_brick', '011_banana',
                       '035_power_drill', '004_sugar_box', '019_pitcher_base',
                       '006_mustard_bottle', '036_wood_block',
                       '009_gelatin_box', '051_large_clamp',
                       '040_large_marker', '003_cracker_box',
                       '025_mug', '052_extra_large_clamp',
                       '021_bleach_cleanser', '002_master_chef_can']

    elif dataset_name == 'FERPlus':
        return ['neutral', 'happiness', 'surprise', 'sadness',
                'anger', 'disgust', 'fear', 'contempt']

    elif dataset_name == 'FER':
        return ['angry', 'disgust', 'fear', 'happy',
                'sad', 'surprise', 'neutral']

    elif dataset_name == 'IMDB':
        return ['man', 'woman']

    elif dataset_name == 'CityScapes':
        return ['void', 'flat', 'construction',
                'object', 'nature', 'sky', 'human', 'vehicle']

    else:
        raise ValueError('Invalid dataset', dataset_name)

    return class_names


def get_arg_to_class(class_names):
    """Constructs dictionary from argument to class names.

    # Arguments
        class_names: List of strings containing the class names.

    # Returns
        Dictionary mapping integer to class name.
    """

    return dict(zip(list(range(len(class_names))), class_names))
