import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA


def put_text(image, text, point, scale, color, thickness):
    """Puts text on image.

    # Arguments
        image: Array of shape `(H, W, 3)`, input image.
        text: String, text to show.
        point: Tuple, coordinate of top corner of text.
        scale: Float, scale of text.
        color: Tuple, RGB color coordinates.
        thickness: Int, text thickness.

    # Returns
        Array: image with text.
    """
    image = cv2.putText(
        image, text, point, FONT, scale, color, thickness, LINE)
    return image


def compute_text_bounds(text, scale, thickness, FONT=FONT):
    """Computes given text size.

    # Arguments
        text: Str, given text.
        scale: Float, text scale.
        thickness: Int, text line thickness.
        FONT: Int, text font.

    # Returns
        Tuple: holding width and height of given text.
    """
    text_size = cv2.getTextSize(text, FONT, scale, thickness)
    return text_size


def draw_opaque_box(image, corner_A, corner_B, color, thickness=-1):
    """ Draws filled rectangle.

    # Arguments
        image: Array of shape `(H, W, 3)`, input image.
        corner_A: List, top left rectangle coordinate.
        corner_B: List, bottom right rectangle coordinate.
        color: List, rectangle's RGB color.
        thickness: Int, rectangle's line thickness.

    # Returns
        Array: image of shape `(H, W, 3)` with filled rectangle.
    """
    image = cv2.rectangle(image, corner_A, corner_B, color, thickness)
    return image


def make_box_transparent(raw_image, image, alpha=0.25):
    """ Blends two images for transparency.

    # Arguments
        raw_image: Array of shape `(H, W, 3)`, first input image.
        image: Array of shape `(H, W, 3)`, second input image.
        alpha: Float, sum weight.

    # Returns
        Array: Blended image of shape `(H, W, 3)`.
    """
    image = cv2.addWeighted(raw_image, 1 - alpha, image, alpha, 0.0)
    return image
