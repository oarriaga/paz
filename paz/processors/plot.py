import numpy as np
import matplotlib.pyplot as plt

from ..abstract import Processor

class PlotErrorCurve(Processor):
    def __init__(self, max_error=0.1, num_steps=10):
        self.max_error = max_error
        self.num_steps = num_steps

    def call(self, errors, title="Title", x_label="x", y_label="y"):
        x_values = np.linspace(0, self.max_error, self.num_steps)
        y_values = list()

        for x_value in x_values:
            y_values.append(np.count_nonzero(errors <= x_value))

        y_values = np.asarray(y_values)
        y_values = y_values/float(len(y_values))

        plt.plot(x_values, y_values)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()