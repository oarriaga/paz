import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA


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
    image = cv2.putText(
        image, text, point, FONT, scale, color, thickness, LINE)
    return image


def get_text_size(text, scale, FONT_THICKNESS, FONT=FONT):
    """Calculates the size of a given text.

    # Arguments
        text: String. Text whose width and height is to be calculated.
        scale: Float. Scale of text.
        FONT_THICKNESS: Integer. Thickness of the lines used for drawing text.
        FONT: Integer. Style of the text font.
    # Returns
        Tuple with shape ((text_W, text_H), baseline)``. The width and height
            of the text
    """
    text_size = cv2.getTextSize(text, FONT, scale, FONT_THICKNESS)
    return text_size


def add_box_border(image, corner_A, corner_B, color, thickness):
    """ Draws an open rectangle from ``corner_A`` to ``corner_B``.

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
    image = cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color), thickness)
    return image


def draw_opaque_box(image, corner_A, corner_B, color, thickness=-1):
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
    image = cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color), thickness)
    return image


def make_box_transparent(raw_image, image, alpha=0.25):
    """ Blends the raw image with bounding box image to add transparency.

    # Arguments
        raw_image: Numpy array of shape ``[H, W, 3]``.
        image: Numpy array of shape ``[H, W, 3]``.
        alpha: Float, weightage parameter of weighted sum.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with rectangle.
    """
    image = cv2.addWeighted(raw_image, 1-alpha, image, alpha, 0.0)
    return image
