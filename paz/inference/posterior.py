from collections import namedtuple
from functools import partial
import pickle

import jax
import jax.numpy as jp
from jax.scipy.stats import gaussian_kde


DEFAULT_PROPERTIES = [
    'scale', 'color', 'ambient', 'diffuse', 'specular', 'shininess', 'classes']
ProgramType = namedtuple('ProgramType', ['trace', 'distributions'])


def Program(trace):
    compute_KDE = partial(gaussian_kde, bw_method='scott')
    distributions = jax.tree_map(compute_KDE, trace)
    return ProgramType(trace, distributions)


def concatenate_leafs(*leafs):
    concatenated_leafs = []
    for leaf in leafs:
        concatenated_leafs.append(leaf)
    return jp.concatenate(concatenated_leafs)


def merge_programs(*programs):
    traces = [program.trace for program in programs]
    trace = jax.tree_map(concatenate_leafs, *traces)
    return Program(trace)


def load_trace(filename):
    return pickle.load(open(filename, 'rb'))


def retrace(trace, fields, frequency=1):
    concept_trace = {}
    for field in fields:
        trace_property = trace[field]
        if len(trace_property.shape) == 2:
            num_chains, num_samples = trace_property.shape
            samples = trace_property.flatten()
            concept_trace[field] = samples[::frequency]
        elif len(trace_property.shape) == 3:
            num_chains, num_samples, num_dimensions = trace_property.shape
            for dimension_arg in range(num_dimensions):
                name = '_'.join([field, str(dimension_arg)])
                samples = trace_property[:, :, dimension_arg].flatten()
                concept_trace[name] = samples[::frequency]
        else:
            raise ValueError('Invalid number of chain dimensions')
    return concept_trace


def unite_dimensions(sample, key, num_dimensions):
    dimension_samples = []
    for dimension_arg in range(num_dimensions):
        dimension_key = '_'.join([key, str(dimension_arg)])
        dimension_samples.append(sample.pop(dimension_key))
    dimension_samples = jp.concatenate(dimension_samples)
    sample[key] = dimension_samples
    return sample


def load_program(filename, frequency=1, fields=DEFAULT_PROPERTIES):
    trace = load_trace(filename)
    trace = retrace(trace, fields, frequency)
    return Program(trace)


def sample_program(key, distributions, num_samples=1):
    samples = sample_distributions(key, num_samples, distributions)
    return jax.tree_map(partial(jp.squeeze, axis=0), samples)


def compute_empirical_KL(p, q, samples):
    return jp.mean(p.logpdf(samples) - q.logpdf(samples), axis=0)


def sample_distributions(key, num_samples, distributions):
    samples = {}
    for name, distribution in distributions.items():
        key, subkey = jax.random.split(key)
        samples[name] = distribution.resample(subkey, (num_samples,))
    return samples


@partial(jax.jit, static_argnums=(3))
def compute_program_distance(key, program_A, program_B, num_samples):
    samples = sample_program(key, program_A.distributions, num_samples)
    total_divergence = 0.0
    LOG2 = jp.log(2)
    for name, p_field in program_A.distributions.items():
        q_field = program_B.distributions[name]
        p_samples = samples[name]
        KL = compute_empirical_KL(p_field, q_field, p_samples)
        total_divergence = total_divergence + (KL / LOG2)
    return total_divergence


def predict(key, protoprograms, program, KL_samples):
    distances = []
    keys = jax.random.split(key, len(protoprograms))
    for subkey, protoprogram in zip(keys, protoprograms):
        distance = compute_program_distance(
            subkey, protoprogram, program, KL_samples)
        distances.append(distance)
    distances = jp.array(distances)
    probabilities = jax.nn.softmax(-distances)
    return probabilities


def render_program(render, samples, shift=None, theta=None):
    # samples = sample_program(key, program.distributions)
    if shift is None:
        samples = unite_dimensions(samples, 'shift', 2)
    else:
        samples['shift'] = shift

    if theta is None:
        samples['theta'] = float(samples['theta'])
    else:
        samples['theta'] = theta
    samples['ambient'] = float(samples['ambient'])
    samples['ambient'] = float(samples['ambient'])
    samples['diffuse'] = float(samples['diffuse'])
    samples['specular'] = float(samples['specular'])
    samples['shininess'] = float(samples['shininess'])
    samples = unite_dimensions(samples, 'scale', 3)
    samples = unite_dimensions(samples, 'color', 3)
    samples = unite_dimensions(samples, 'classes', 3)
    image, depth = render(samples)
    return image, depth
