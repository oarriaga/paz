from paz import processors as pr

from draw import (add_box_border, draw_opaque_box, get_text_size,
                  make_box_transparent, put_text)


class DrawBoxes2D(pr.DrawBoxes2D):
    """Draws bounding boxes from Boxes2D messages.

    # Arguments
        class_names: List, indicating class names.
        colors: List holding color values.
        weighted: Bool, denoting bounding box color to be weighted.
        scale: Float. Scale of text drawn.
        with_score: Bool, denoting if confidence be shown.
    """
    def __init__(
            self, class_names=None, colors=None,
            weighted=False, scale=0.7, with_score=True):
        super().__init__(
            class_names, colors, weighted, scale, with_score)

    def compute_prediction_parameters(self, box2D):
        x_min, y_min, x_max, y_max = box2D.coordinates
        class_name = box2D.class_name
        color = self.class_to_color[class_name]
        if self.weighted:
            color = [int(channel * box2D.score) for channel in color]
        if self.with_score:
            text = '{} :{}%'.format(class_name, round(box2D.score * 100))
        if not self.with_score:
            text = '{}'.format(class_name)
        return x_min, y_min, x_max, y_max, color, text

    def call(self, image, boxes2D):
        raw_image = image.copy()
        for box2D in boxes2D:
            prediction_parameters = self.compute_prediction_parameters(box2D)
            x_min, y_min, x_max, y_max, color, text = prediction_parameters
            draw_opaque_box(image, (x_min, y_min), (x_max, y_max), color)
        image = make_box_transparent(raw_image, image)
        for box2D in boxes2D:
            prediction_parameters = self.compute_prediction_parameters(box2D)
            x_min, y_min, x_max, y_max, color, text = prediction_parameters
            add_box_border(image, (x_min, y_min), (x_max, y_max), color, 2)
            text_size = get_text_size(text, self.scale, 1)
            (text_W, text_H), _ = text_size
            args = (image, (x_min + 2, y_min + 2),
                    (x_min + text_W + 5, y_min + text_H + 5), (255, 174, 66))
            draw_opaque_box(*args)
            args = (image, text, (x_min + 2, y_min + 17), self.scale,
                    (0, 0, 0), 1)
            put_text(*args)
        return image
