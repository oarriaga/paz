from paz import processors as pr
from paz.backend.image.draw import draw_rectangle
from draw import (compute_text_bounds, draw_opaque_box, make_box_transparent,
                  put_text)


class DrawBoxes2D(pr.DrawBoxes2D):
    """Draws bounding boxes from Boxes2D messages.

    # Arguments
        class_names: List, class names.
        colors: List, color values.
        weighted: Bool, whether to weight bounding box color.
        scale: Float. Scale of text drawn.
        with_score: Bool, denoting if confidence be shown.
    """
    def __init__(
            self, class_names=None, colors=None,
            weighted=False, scale=0.7, with_score=True):
        super().__init__(
            class_names, colors, weighted, scale, with_score)

    def compute_box_color(self, box2D):
        class_name = box2D.class_name
        color = self.class_to_color[class_name]
        if self.weighted:
            color = [int(channel * box2D.score) for channel in color]
        return color

    def compute_text(self, box2D):
        class_name = box2D.class_name
        text = '{}'.format(class_name)
        if self.with_score:
            text = '{} :{}%'.format(class_name, round(box2D.score * 100))
        return text

    def get_text_box_parameters(self):
        thickness = 1
        offset_x = 2
        offset_y = 17
        color = (0, 0, 0)
        text_parameters = [thickness, offset_x, offset_y, color]
        box_start_offset = 2
        box_end_offset = 5
        box_color = (255, 174, 66)
        text_box_parameters = [box_start_offset, box_end_offset, box_color]
        return [text_box_parameters, text_parameters]

    def call(self, image, boxes2D):
        raw_image = image.copy()
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates
            color = self.compute_box_color(box2D)
            draw_opaque_box(image, (x_min, y_min), (x_max, y_max), color)
        image = make_box_transparent(raw_image, image)
        text_box_parameters, text_parameters = self.get_text_box_parameters()
        offset_start, offset_end, text_box_color = text_box_parameters
        text_thickness, offset_x, offset_y, text_color = text_parameters
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates
            color = self.compute_box_color(box2D)
            draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            text = self.compute_text(box2D)
            text_size = compute_text_bounds(text, self.scale, text_thickness)
            (text_W, text_H), _ = text_size
            args = (image, (x_min + offset_start, y_min + offset_start),
                    (x_min + text_W + offset_end, y_min + text_H + offset_end),
                    text_box_color)
            draw_opaque_box(*args)
            args = (image, text, (x_min + offset_x, y_min + offset_y),
                    self.scale, text_color, text_thickness)
            put_text(*args)
        return image
