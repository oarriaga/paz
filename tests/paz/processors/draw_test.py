import pytest
from paz import processors as pr


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
