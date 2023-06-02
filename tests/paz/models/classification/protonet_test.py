import pytest
import numpy as np
from keras.utils.layer_utils import count_params
from paz.models.classification.protonet import (
    ProtoEmbedding, ProtoNet, compute_pairwise_distances)


def parametrize(Model, model_args, output_shape, params, static_params):
    """Builds test for model with a given parametrization

    # Arguments:
        Model: Keras model.
        model_args: List with required model arguments for Model instantiation.
        output_shape: List with expected model output shapes.
        params: List of integers with `trainable` parameters.
        static_params: List of integers with number of `non_trainable` params.

    # Returns:
        Parametrized test function.
    """
    parametrization = []
    for args in zip(output_shape, params, static_params):
        parametrization.append(args)
    function_arguments = 'outro_shape, parameters, static_parameters'

    @pytest.mark.parametrize(function_arguments, parametrization)
    def test(outro_shape, parameters, static_parameters):
        model = Model(*model_args)
        model_params = count_params(model.trainable_weights)
        model_static_params = count_params(model.non_trainable_weights)
        assert model_params == parameters
        assert model_static_params == static_parameters
        assert model.output_shape == outro_shape
        del model
    return test


test_embedding = parametrize(ProtoEmbedding,
                             [(28, 28, 1), 4],
                             [(None, 64)], [111_936], [512])

test_protonet = parametrize(
    ProtoNet,
    [ProtoEmbedding((28, 28, 1), 4), 20, 5, 1, (28, 28, 1)],
    [(20, 20)], [111_936], [512])


def test_pairwise_distances_zero():
    a = np.random.rand(10, 32)
    distances = compute_pairwise_distances(a, a).numpy()
    assert np.allclose(np.diag(distances), 0.0)
    assert distances.shape == (10, 10)


def test_pairwise_distances_positivity():
    a = np.random.rand(7, 100)
    b = np.random.rand(8, 100)
    distances = compute_pairwise_distances(a, b).numpy()
    assert np.alltrue(distances > 0)
    assert distances.shape == (7, 8)


def test_pairwise_distances_symmetry():
    a = np.random.rand(30, 5)
    b = np.random.rand(15, 5)
    distances_AB = compute_pairwise_distances(a, b).numpy()
    distances_BA = compute_pairwise_distances(b, a).numpy()
    assert np.allclose(distances_AB, distances_BA.T)
    assert distances_AB.shape == (30, 15)
    assert distances_BA.shape == (15, 30)


def test_pairwise_distances_orthogonal():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    distances = compute_pairwise_distances(a, b).numpy()
    assert np.allclose(distances, 0)


def test_pairwise_distances_with_values_to_zero():
    a = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    values = compute_pairwise_distances(a, b).numpy()
    distances = np.array([[1.0], [1.0 / 3.0], [2.0 / 3.0]])
    assert np.allclose(values, distances)


def test_pairwise_distances_with_values_to_negative():
    a = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    b = np.array([[-1.0, -1.0, -1.0]])
    values = compute_pairwise_distances(a, b).numpy()
    distances = np.array([[4.0], [2.0], [3.0]])
    assert np.allclose(values, distances)
