import tensorflow.compat.v1 as tf
import numpy as np
import collections
import six
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


coco = {
    # 0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua',
    'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood',
    'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue',
    'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey',
    'DarkKhaki', 'DarkOrange', 'DarkOrchid',
    'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise',
    'DarkViolet', 'DeepPink', 'DeepSkyBlue',
    'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro',
    'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew',
    'HotPink', 'IndianRed', 'Ivory',
    'Khaki', 'Lavender', 'LavenderBlush',
    'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow',
    'LightGray', 'LightGrey', 'LightGreen',
    'LightPink', 'LightSalmon', 'LightSeaGreen',
    'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey',
    'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta',
    'MediumAquaMarine', 'MediumOrchid', 'MediumPurple',
    'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream',
    'MistyRose', 'Moccasin', 'NavajoWhite',
    'OldLace', 'Olive', 'OliveDrab',
    'Orange', 'OrangeRed', 'Orchid',
    'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise',
    'PaleVioletRed', 'PapayaWhip', 'PeachPuff',
    'Peru', 'Pink', 'Plum',
    'PowderBlue', 'Purple', 'Red',
    'RosyBrown', 'RoyalBlue', 'SaddleBrown',
    'Green', 'SandyBrown', 'SeaGreen',
    'SeaShell', 'Sienna', 'Silver',
    'SkyBlue', 'SlateBlue', 'SlateGray',
    'SlateGrey', 'Snow', 'SpringGreen',
    'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato',
    'Turquoise', 'Violet', 'Wheat',
    'White', 'WhiteSmoke', 'Yellow',
    'YellowGreen'
]


def _get_multiplier_for_color_randomness():
    """Returns a multiplier to get semi-random colors from successive indices.

    This function computes a prime number, p, in the range [2, 17] that:
    - is closest to len(STANDARD_COLORS) / 10
    - does not divide len(STANDARD_COLORS)

    If no prime numbers in that range
    satisfy the constraints, p is returned as 1.

    Once p is established, it can be used as a multiplier to select
    non-consecutive colors from STANDARD_COLORS:
    colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
    """
    num_colors = len(STANDARD_COLORS)
    prime_candidates = [5, 7, 11, 13, 17]

    # Remove all prime candidates that divide the number of colors.
    prime_candidates = [p for p in prime_candidates if num_colors % p]
    if not prime_candidates:
        return 1

    # Return the closest prime number to num_colors / 10.
    abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
    num_candidates = len(abs_distance)
    inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
    return prime_candidates[inds[0]]


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=False):
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
    (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
    ymin, xmin, ymax, xmax as relative to the image.
    Otherwise treat coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=False):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified
    in either absolute (pixel) or
    normalized coordinates by setting the
    use_normalized_coordinates argument.

    Each string in display_str_list is displayed
    on a separate line above the
    bounding box in black text on a rectangle
    filled with the input 'color'.
    If the top of the bounding box extends to
    the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
    (each to be shown on its own line).
    use_normalized_coordinates: If True (default),
    treat coordinates ymin, xmin, ymax, xmax as relative
    to the image. Otherwise treat coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display
    # strings added to the top of the bounding
    # box exceeds the top of the image, stack
    # the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def visualize_boxes_and_labels_on_image_array(image,
                                              boxes,
                                              classes,
                                              scores,
                                              category_index,
                                              track_ids=None,
                                              use_normalized_coordinates=False,
                                              max_boxes_to_draw=20,
                                              min_score_thresh=.5,
                                              agnostic_mode=False,
                                              line_thickness=4,
                                              skip_boxes=False,
                                              skip_scores=False,
                                              skip_labels=False,
                                              skip_track_ids=False):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].numpy().tolist())
            display_str = ''
            if not skip_labels:
                if not agnostic_mode:
                    if classes[i].numpy() in six.viewkeys(category_index):
                        class_name = category_index[classes[i].numpy()]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
            if not skip_scores:
                if not display_str:
                    display_str = '{}%'.format(int(100 * scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str,
                                                   int(100 * scores[i]))
            if not skip_track_ids and track_ids is not None:
                if not display_str:
                    display_str = 'ID {}'.format(track_ids[i])
                else:
                    display_str = '{}: ID {}'.format(display_str, track_ids[i])
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = STANDARD_COLORS[
                classes[i] % len(STANDARD_COLORS)]
    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=0 if skip_boxes else line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
    return image


def visualize_image_prediction(image, prediction):
    boxes = prediction[:, 1:5]
    classes = tf.cast(prediction[:, 6], dtype=tf.int32)
    scores = prediction[:, 5]
    return visualize_image(image, boxes, classes, scores)


def visualize_image(image,
                    boxes,
                    classes,
                    scores,
                    min_score_thresh=0.4,
                    max_boxes_to_draw=1000,
                    line_thickness=2):
    label_map = coco
    category_index = {k: {'id': k, 'name': label_map[k]} for k in label_map}
    img = np.array(image)
    img = visualize_boxes_and_labels_on_image_array(
        img,
        boxes,
        classes,
        scores,
        category_index,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw,
        line_thickness=line_thickness)
    return img
