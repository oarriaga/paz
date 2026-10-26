import pytest
import numpy as np
from paz import processors as pr
from paz.abstract import Box2D


def test_DrawBoxes2D_with_invalid_class_names_type():
    with pytest.raises(TypeError):
        class_names = 'Face'
        colors = [[255, 0, 0]]
        pr.DrawBoxes2D(class_names, colors)


def test_DrawBoxes2D_with_invalid_colors_type():
    with pytest.raises(TypeError):
        class_names = ['Face']
        colors = [255, 0, 0]
        pr.DrawBoxes2D(class_names, colors)


def test_DrawBoxes2D_with_fractional_coordinates():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes2D = [Box2D([10.4, 12.6, 40.5, 50.2], 0.9, 'Face')]
    draw_boxes2D = pr.DrawBoxes2D(['Face'], [[255, 0, 0]])
    image = draw_boxes2D(image, boxes2D)
    assert np.all(image[13, 10] == [255, 0, 0])
    assert np.all(image[50, 40] == [255, 0, 0])
