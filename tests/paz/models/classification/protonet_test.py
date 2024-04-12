import pytest
import numpy as np
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


def count_params(weights):
    """Count the total number of scalars composing the weights.
    This function is taken from the repository of [Keras]
    (https://github.com/keras-team/keras/blob/428ed9f03a0a0b2edc22d4ce29
     001857f617227c/keras/utils/layer_utils.py#L107)
    This is a patch and it should be removed eventually.

    # Arguments:
        weights: List, containing the weights
            on which to compute params.

    # Returns:
        Int, the total number of scalars composing the weights.
    """
    unique_weights = {id(w): w for w in weights}.values()
    unique_weights = [w for w in unique_weights if hasattr(w, "shape")]
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))
