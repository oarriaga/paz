import pytest
from keras.utils.layer_utils import count_params
from protonet import Embedding, PROTONET


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


test_embedding = parametrize(Embedding,
                             [(28, 28, 1), 4],
                             [(None, 64)], [111_936], [512])

test_protonet = parametrize(PROTONET,
                            [Embedding((28, 28, 1), 4), 20, 5, 1, (28, 28, 1)],
                            [(20, 20)], [111_936], [512])
