import unittest
from paz import processors as pr


class TestDrawBoxes2D(unittest.TestCase):
    """Test the type of the parameters for DrawBoxes2D class.
        class_names: List of strings.
        colors: List of lists containing the color values
    """
    def test_parameters_type(self):
        # class_name is not of required type
        self.class_names = 'Face'
        self.colors = [[255, 0, 0]]
        with self.assertRaises(TypeError) as context:
            pr.DrawBoxes2D(self.class_names, self.colors)
        self.assertEqual("Class name should be of type 'List of strings'",
                         str(context.exception))

        # colors is not of required type
        self.class_names = ['Face']
        self.colors = [255, 0, 0]
        with self.assertRaises(TypeError) as context:
            pr.DrawBoxes2D(self.class_names, self.colors)
        self.assertEqual("Colors should be of type 'List of lists'",
                         str(context.exception))


if __name__ == '__main__':
    unittest.main()
