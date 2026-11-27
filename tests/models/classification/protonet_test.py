import numpy as np
from paz.models.classification.protonet import ProtoEmbedding, ProtoNet
from paz.layers.compute_pairwise_distances import compute_pairwise_distances


def count_parameters(weights):
    return sum(int(np.prod(weight.shape)) for weight in weights)


def test_embedding_has_the_published_size():
    model = ProtoEmbedding((28, 28, 1), 4)
    assert model.output_shape == (None, 64)
    assert count_parameters(model.trainable_weights) == 111_936
    assert count_parameters(model.non_trainable_weights) == 512


def test_protonet_assigns_support_images_to_their_own_class():
    num_classes, image_shape = 5, (28, 28, 1)
    embedding = ProtoEmbedding(image_shape, 4)
    model = ProtoNet(embedding, num_classes, 1, 1, image_shape)
    images = np.random.default_rng(0).random((num_classes, 1, *image_shape))
    scores = np.asarray(model([images, images]))
    assert scores.shape == (num_classes, num_classes)
    assert np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(scores.argmax(axis=1), np.arange(num_classes))


def test_pairwise_distances_zero():
    a = np.random.rand(10, 32)
    distances = np.asarray(compute_pairwise_distances(a, a))
    assert np.allclose(np.diag(distances), 0.0)
    assert distances.shape == (10, 10)


def test_pairwise_distances_positivity():
    a = np.random.rand(7, 100)
    b = np.random.rand(8, 100)
    distances = np.asarray(compute_pairwise_distances(a, b))
    assert np.all(distances > 0)
    assert distances.shape == (7, 8)


def test_pairwise_distances_symmetry():
    a = np.random.rand(30, 5)
    b = np.random.rand(15, 5)
    distances_AB = np.asarray(compute_pairwise_distances(a, b))
    distances_BA = np.asarray(compute_pairwise_distances(b, a))
    assert np.allclose(distances_AB, distances_BA.T)
    assert distances_AB.shape == (30, 15)
    assert distances_BA.shape == (15, 30)


def test_pairwise_distances_with_values_to_zero():
    a = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    values = np.asarray(compute_pairwise_distances(a, b))
    distances = np.array([[1.0], [1.0 / 3.0], [2.0 / 3.0]])
    assert np.allclose(values, distances)


def test_pairwise_distances_with_values_to_negative():
    a = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    b = np.array([[-1.0, -1.0, -1.0]])
    values = np.asarray(compute_pairwise_distances(a, b))
    distances = np.array([[4.0], [2.0], [3.0]])
    assert np.allclose(values, distances)

