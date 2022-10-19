import cv2

GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
FILLED = cv2.FILLED


def put_text(image, text, point, scale, color, thickness):
    """Draws text in image.

    # Arguments
        image: Numpy array.
        text: String. Text to be drawn.
        point: Tuple of coordinates indicating the top corner of the text.
        scale: Float. Scale of text.
        color: Tuple of integers. RGB color coordinates.
        thickness: Integer. Thickness of the lines used for drawing text.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with text.
    """
    # cv2.putText returns an image in contrast to other drawing cv2 functions.
    return cv2.putText(image, text, point, FONT, scale, color, thickness, LINE)


def get_text_size(text, scale, FONT_THICKNESS, FONT=FONT):
    return cv2.getTextSize(text, FONT, scale, FONT_THICKNESS)


def add_box_border(image, corner_A, corner_B, color, thickness):
    """ Draws a filled rectangle from ``corner_A`` to ``corner_B``.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        corner_A: List of length two indicating ``(y, x)`` openCV coordinates.
        corner_B: List of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer/openCV Flag. Thickness of rectangle line.
            or for filled use cv2.FILLED flag.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with rectangle.
    """
    return cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color),
        thickness)


def draw_opaque_box(image, corner_A, corner_B, color, thickness=-1):
    return cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color),
        thickness)


def make_box_transparent(raw_image, image, alpha=0.30):
    return cv2.addWeighted(raw_image, 1-alpha, image, alpha, 0.0)
