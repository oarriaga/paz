import numpy as np
import jax
import jax.numpy as jp
import pytest

from paz.models.foundation.sam2 import hiera, image_encoder
from paz.models.foundation.sam2 import prompt_encoder, mask_decoder
from paz.models.foundation.sam2 import model as sam2_model, predict
from paz.models.foundation.sam2 import preprocessing
from paz.models.foundation.sam2.windows import window_partition
from paz.models.foundation.sam2.windows import window_unpartition
from paz.models.foundation.sam2.configuration import TINY, backbone_channels


def test_preprocess_shape_and_normalization():
    image = np.full((30, 40, 3), 255, np.uint8)
    output = np.array(preprocessing.preprocess_image(image))
    assert output.shape == (1024, 1024, 3)
    expected = (1.0 - np.array(preprocessing.MEAN)) / preprocessing.STDV
    assert np.allclose(output[512, 512], expected, atol=1e-4)


def test_transform_coords_maps_to_resolution():
    coords = preprocessing.transform_coords([[50.0, 25.0]], (100, 200))
    assert np.allclose(np.array(coords), [[256.0, 256.0]])


def test_transform_boxes_reshapes_to_corners():
    boxes = preprocessing.transform_boxes([0.0, 0.0, 200.0, 100.0], (100, 200))
    assert np.array(boxes).shape == (1, 2, 2)
    assert np.allclose(np.array(boxes)[0, 1], [1024.0, 1024.0])


def test_window_partition_unpartition_roundtrip():
    x = jp.asarray(np.random.RandomState(0).randn(2, 20, 28, 5), jp.float32)
    windows, padded = window_partition(x, 8)
    assert windows.shape[1:] == (8, 8, 5)
    restored = window_unpartition(windows, 8, padded, (20, 28))
    assert np.allclose(np.array(restored), np.array(x), atol=1e-6)


def test_bicubic_matrix_rows_sum_to_one():
    matrix = hiera.bicubic_resize_matrix(7, 64)
    assert matrix.shape == (64, 7)
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5)


def test_block_specifications_match_config():
    specifications, stage_ends = hiera.build_block_specifications(TINY)
    assert len(specifications) == sum(TINY.stages)
    assert stage_ends == [0, 2, 9, 11]
    pooled = [index for index, *_ , pool, _ in specifications if pool]
    assert pooled == [1, 3, 10]


def test_image_encoder_output_shapes():
    model = image_encoder.build(TINY)
    embedding, high_res_0, high_res_1 = model.outputs
    assert tuple(embedding.shape) == (None, 64, 64, 256)
    assert tuple(high_res_0.shape) == (None, 256, 256, 32)
    assert tuple(high_res_1.shape) == (None, 128, 128, 64)


def test_backbone_channels_follow_embed_dim():
    assert backbone_channels(TINY) == [768, 384, 192, 96]


def test_point_encoder_sparse_shape():
    model = prompt_encoder.build_points()
    coords = np.zeros((1, 3, 2), np.float32)
    labels = np.zeros((1, 3), np.float32)
    sparse = np.array(model((coords, labels)))
    assert sparse.shape == (1, 3, 256)


def test_point_label_semantics():
    model = prompt_encoder.build_points()
    layer = model.get_layer("point_label_embed")
    corners = np.arange(4 * 256, dtype="float32").reshape(4, 256)
    not_a_point = np.full((1, 256), -7.0, "float32")
    layer.set_weights([corners, not_a_point])
    coords = np.zeros((1, 3, 2), np.float32)
    labels = np.array([[0, 1, -1]], np.float32)
    sparse = np.array(model((coords, labels)))
    difference = sparse[0, 1] - sparse[0, 0]
    assert np.allclose(difference, corners[1] - corners[0], atol=1e-4)
    assert np.allclose(sparse[0, 2], not_a_point[0], atol=1e-4)


def test_mask_downscaling_shape():
    model = prompt_encoder.build_mask_downscaling()
    dense, no_mask = model(np.zeros((1, 256, 256, 1), np.float32))
    assert np.array(dense).shape == (1, 64, 64, 256)
    assert np.array(no_mask).shape == (1, 64, 64, 256)


def test_mask_decoder_shapes():
    model = mask_decoder.build()
    inputs = [np.zeros((1, 64, 64, 256), np.float32),
              np.zeros((1, 256, 256, 32), np.float32),
              np.zeros((1, 128, 128, 64), np.float32),
              np.zeros((1, 2, 256), np.float32),
              np.zeros((1, 64, 64, 256), np.float32),
              np.zeros((1, 64, 64, 256), np.float32)]
    masks, iou, obj = model(inputs)
    assert np.array(masks).shape == (1, 4, 256, 256)
    assert np.array(iou).shape == (1, 4)
    assert np.array(obj).shape == (1, 1)


@pytest.fixture(scope="module")
def bundle():
    return sam2_model.build(TINY)


def test_single_and_multimask_selection(bundle):
    image = np.zeros((80, 120, 3), np.uint8)
    features = predict.encode_image(bundle, image)
    multi, scores, _ = predict.predict(
        bundle, features, points=[[60.0, 40.0]], labels=[1], multimask=True)
    single, _, _ = predict.predict(
        bundle, features, points=[[60.0, 40.0]], labels=[1], multimask=False)
    assert np.array(multi).shape == (1, 3, 80, 120)
    assert np.array(single).shape == (1, 1, 80, 120)
    assert np.array(scores).shape == (1, 3)


def test_box_and_point_box_shapes(bundle):
    image = np.zeros((80, 120, 3), np.uint8)
    features = predict.encode_image(bundle, image)
    box, _, _ = predict.predict(bundle, features, box=[10.0, 10.0, 60.0, 50.0])
    combined, _, _ = predict.predict(
        bundle, features, points=[[30.0, 30.0]], labels=[1],
        box=[10.0, 10.0, 60.0, 50.0])
    assert np.array(box).shape == (1, 3, 80, 120)
    assert np.array(combined).shape == (1, 3, 80, 120)


def test_window_partition_jit_cache_is_stable():
    partition = jax.jit(lambda x: window_partition(x, 8)[0])
    for seed in range(3):
        data = jp.asarray(np.random.RandomState(seed).randn(1, 16, 16, 4))
        partition(data)
    assert partition._cache_size() == 1
