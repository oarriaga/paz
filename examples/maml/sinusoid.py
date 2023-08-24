import numpy as np


def build_equally_spaced_points(num_points, min_x, max_x):
    return np.linspace(min_x, max_x, num_points)


def sample_random_points(RNG, num_points, min_x, max_x):
    return RNG.uniform(min_x, max_x, num_points)


def sample_amplitude(RNG, min_amplitude=0.1, max_amplitude=5.0):
    return RNG.uniform(min_amplitude, max_amplitude)


def sample_phase(RNG):
    return RNG.uniform(0, np.pi)


def compute_sinusoid(x, amplitude, phase):
    return amplitude * np.sin(x - phase)


def Sinusoid(RNG, num_points, min_amplitude=0.1, max_amplitude=5.0,
             min_x=-5.0, max_x=5.0):
    amplitude = sample_amplitude(RNG, min_amplitude, max_amplitude)
    phase = sample_phase(RNG)

    def sample(batch_size=None, equally_spaced=False):
        batch_size = num_points if batch_size is None else batch_size
        if equally_spaced:
            x_support = build_equally_spaced_points(batch_size, min_x, max_x)
            x_queries = build_equally_spaced_points(batch_size, min_x, max_x)
        else:
            x_support = sample_random_points(RNG, batch_size, min_x, max_x)
            x_queries = sample_random_points(RNG, batch_size, min_x, max_x)
        y_support = compute_sinusoid(x_support, amplitude, phase)
        y_queries = compute_sinusoid(x_queries, amplitude, phase)
        return ((x_support, y_support), (x_queries, y_queries))
    return sample
