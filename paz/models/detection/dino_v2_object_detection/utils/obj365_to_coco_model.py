import keras.ops as k

# COCO category IDs (1-indexed, non-contiguous: 80 categories)
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
# Corresponding Objects365 category indices (0-indexed)
OBJ365_IDS = [
    0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249,
    24, 56, 139, 92, 78, 99, 96, 144, 295, 178, 180, 38, 39,
    13, 43, 120, 219, 148, 173, 165, 154, 137, 113, 145, 146, 204,
    8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152,
    234, 143, 150, 97, 2, 50, 25, 75, 98, 153, 37, 73, 115,
    132, 106, 61, 163, 134, 277, 81, 133, 18, 94, 30, 169, 70,
    328, 226,
]


def remap_obj365_rows(cur_weights, pretrain_weights):
    # cur_weights[coco_id] = pretrain_weights[obj365_id + 1]
    remapped = k.convert_to_numpy(cur_weights).copy()
    pretrained = k.convert_to_numpy(pretrain_weights)
    for coco_id, obj365_id in zip(COCO_IDS, OBJ365_IDS):
        remapped[coco_id] = pretrained[obj365_id + 1]
    return remapped


def get_coco_pretrain_from_obj365(cur_weights, pretrain_weights):
    cur_shape = tuple(k.shape(cur_weights))
    pretrain_shape = tuple(k.shape(pretrain_weights))
    # Matching shapes mean the head already has COCO layout.
    if cur_shape == pretrain_shape:
        weights = pretrain_weights
    else:
        weights = remap_obj365_rows(cur_weights, pretrain_weights)
    return weights
