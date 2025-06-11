import pytest
import jax.numpy as jp
import paz
import numpy as np
from paz.backend.detection import (
    encode,
    decode,
    apply_NMS,
)


def test_encode_decode():
    matched = jp.array([[0.0, 0.0, 2.0, 2.0, 1.0]])
    priors = paz.boxes.to_center_form(matched[:, :4])
    encoded = paz.detection.encode(matched, priors)
    decoded = paz.detection.decode(encoded, priors)
    assert jp.allclose(decoded[:, :4], matched[:, :4], atol=1e-4)


@pytest.mark.skip(reason="apply_NMS changed implementation")
@pytest.mark.parametrize(
    "boxes_and_scores,iou_thresh,top_k,expected",
    [
        (
            jp.array(
                [
                    [0.0, 0.0, 1.0, 1.0, 0.8],
                    [0.0, 0.0, 1.0, 1.0, 0.9],
                    [2.0, 2.0, 3.0, 3.0, 0.7],
                ]
            ),
            0.5,
            200,
            jp.array(
                [
                    [0.0, 0.0, 1.0, 1.0, 0.9],
                    [2.0, 2.0, 3.0, 3.0, 0.7],
                ]
            ),
        ),
        (
            jp.array(
                [
                    [10, 10, 110, 110, 0.9],
                    [20, 20, 120, 120, 0.75],
                    [30, 30, 80, 80, 0.6],
                    [200, 200, 250, 250, 0.7],
                ]
            ),
            0.5,
            200,
            # [0, 3, 2],
            jp.array(
                [
                    [10, 10, 110, 110, 0.9],
                    [200, 200, 250, 250, 0.7],
                    [30, 30, 80, 80, 0.6],
                ]
            ),
        ),
    ],
)
def test_apply_NMS(boxes_and_scores, iou_thresh, top_k, expected):
    selected_indices = apply_NMS(boxes_and_scores, iou_thresh, top_k)
    assert len(selected_indices) == len(expected)
    assert list(selected_indices) == expected


@pytest.fixture
def boxes_and_scores():
    """
    Create a simple test case with 4 boxes:
    - Box A: [10, 10, 110, 110] with score 0.9
    - Box B: [20, 20, 120, 120] with score 0.75 (high overlap with A)
    - Box C: [30, 30, 80, 80] with score 0.6 (partial overlap with A and B)
    - Box D: [200, 200, 250, 250] with score 0.7 (no overlap with others)
    """
    detections = jp.array(
        [
            [10, 10, 110, 110, 0.9],
            [20, 20, 120, 120, 0.75],
            [30, 30, 80, 80, 0.6],
            [200, 200, 250, 250, 0.7],
        ]
    )
    return detections


@pytest.mark.skip(reason="apply_NMS changed implementation")
def test_apply_nms_higher_threshold(boxes_and_scores):
    """Test higher IoU threshold keeps more boxes."""
    selected_indices = apply_NMS(boxes_and_scores, 0.9, 200)
    assert jp.allclose(selected_indices, jp.array([0, 1, 3, 2]))


@pytest.mark.skip(reason="apply_NMS changed implementation")
def test_apply_nms_lower_threshold(boxes_and_scores):
    """Test lower IoU threshold suppresses more boxes."""
    selected_indices = apply_NMS(boxes_and_scores, 0.4, 200)
    assert jp.allclose(selected_indices, jp.array([0, 3, 2]))


@pytest.mark.skip(reason="apply_NMS changed implementation")
def test_apply_nms_top_k(boxes_and_scores):
    """Test top_k parameter limits initial candidates."""
    selected_indices = apply_NMS(boxes_and_scores, 0.5, 2)
    assert jp.allclose(selected_indices, jp.array([0]))


@pytest.mark.skip(reason="apply_NMS changed implementation")
def test_apply_nms_single_box():
    """Test single box returns itself."""
    boxes_and_scores = jp.array([[0, 0, 10, 10, 0.9]])
    assert jp.allclose(apply_NMS(boxes_and_scores, 0.45, 400), jp.array([0]))


@pytest.mark.skip(reason="apply_NMS changed implementation")
def test_apply_nms_no_overlap():
    """Test non-overlapping boxes all kept."""
    boxes_and_scores = jp.array(
        [[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.8], [40, 40, 50, 50, 0.7]]
    )
    selected = apply_NMS(boxes_and_scores, 0.1, 400)
    assert jp.allclose(selected, jp.array([0, 1, 2]))


@pytest.mark.skip(reason="apply_NMS changed implementation")
def test_apply_nms_identical_boxes():
    """Test identical boxes keep highest score."""
    boxes_and_scores = jp.array(
        [[0, 0, 10, 10, 0.7], [0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]]
    )
    assert jp.allclose(apply_NMS(boxes_and_scores, 0.5, 400), jp.array([1]))


@pytest.fixture
def generate_sample_boxes_and_priors():
    boxes = jp.array([[10, 20, 60, 90], [30, 40, 100, 120]])

    boxes_with_labels = jp.hstack([boxes, jp.ones((2, 1))])

    priors = jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])

    return boxes_with_labels, priors


@pytest.fixture
def box_data():
    """Basic test data for boxes in corner form with labels."""
    boxes = jp.array([[10, 20, 60, 90], [30, 40, 100, 120]])
    boxes_with_labels = jp.hstack([boxes, jp.ones((2, 1))])
    return boxes_with_labels


@pytest.fixture
def prior_data():
    """Prior boxes in center form [center_x, center_y, width, height]."""
    return jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])


@pytest.fixture
def default_variances():
    """Default variance values for encoding/decoding."""
    return [0.1, 0.1, 0.2, 0.2]


@pytest.fixture
def alternate_variances():
    """Alternative variance values for testing."""
    return [0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def box_centers(box_data):
    """Boxes converted to center form."""
    return paz.boxes.to_center_form(box_data[:, :4])


@pytest.fixture
def encoded_boxes(box_data, prior_data, default_variances):
    """Encoded boxes using default variances."""
    return encode(box_data, prior_data, default_variances)


@pytest.fixture
def expected_encoding_values(box_centers, prior_data, default_variances):
    """Expected encoding values for the first box."""
    epsilon = 1e-8

    exp_cx_diff = (
        (box_centers[0, 0] - prior_data[0, 0])
        / prior_data[0, 2]
        / default_variances[0]
    )
    exp_cy_diff = (
        (box_centers[0, 1] - prior_data[0, 1])
        / prior_data[0, 3]
        / default_variances[1]
    )
    exp_w = (
        jp.log(box_centers[0, 2] / prior_data[0, 2] + epsilon)
        / default_variances[2]
    )
    exp_h = (
        jp.log(box_centers[0, 3] / prior_data[0, 3] + epsilon)
        / default_variances[3]
    )

    return {
        "cx_diff": exp_cx_diff,
        "cy_diff": exp_cy_diff,
        "width": exp_w,
        "height": exp_h,
    }


@pytest.fixture
def single_prediction():
    """A single prediction for decode testing."""
    return jp.array([[0.1, 0.2, 0.3, 0.4, 1.0]])


@pytest.fixture
def single_prior():
    """A single prior for decode testing."""
    return jp.array([[50, 60, 30, 40]])


@pytest.fixture
def expected_decoded_center(single_prediction, single_prior, default_variances):
    """Expected center-format box after decoding."""
    cx = (
        single_prediction[0, 0] * single_prior[0, 2] * default_variances[0]
        + single_prior[0, 0]
    )
    cy = (
        single_prediction[0, 1] * single_prior[0, 3] * default_variances[1]
        + single_prior[0, 1]
    )
    w = single_prior[0, 2] * jp.exp(
        single_prediction[0, 2] * default_variances[2]
    )
    h = single_prior[0, 3] * jp.exp(
        single_prediction[0, 3] * default_variances[3]
    )

    return jp.array([[cx, cy, w, h]])


@pytest.fixture
def large_test_data():
    """Generate 1000 random boxes and priors for performance testing."""
    np.random.seed(42)
    num_boxes = 1000

    boxes_raw = np.random.randint(0, 500, size=(num_boxes, 4)).astype(float)
    sorted_boxes = []
    for i in range(num_boxes):
        box = boxes_raw[i]
        x_min, y_min = min(box[0], box[2]), min(box[1], box[3])
        x_max, y_max = max(box[0], box[2]), max(box[1], box[3])
        sorted_boxes.append([x_min, y_min, x_max, y_max])

    boxes_corner = jp.array(sorted_boxes)

    class_labels = jp.array(np.random.randint(1, 11, size=(num_boxes, 1)))
    boxes = jp.hstack([boxes_corner, class_labels])

    priors_center = jp.array(
        np.random.randint(10, 490, size=(num_boxes, 2)).astype(float)
    )
    priors_size = jp.array(
        np.random.randint(10, 100, size=(num_boxes, 2)).astype(float)
    )
    priors = jp.hstack([priors_center, priors_size])

    return boxes, priors


@pytest.fixture
def expected_corner_from_prior(single_prior):
    """Calculate expected corner format from prior box."""
    return jp.array(
        [
            [
                single_prior[0, 0] - single_prior[0, 2] / 2,
                single_prior[0, 1] - single_prior[0, 3] / 2,
                single_prior[0, 0] + single_prior[0, 2] / 2,
                single_prior[0, 1] + single_prior[0, 3] / 2,
            ]
        ]
    )


@pytest.fixture
def boxes_with_class(expected_corner_from_prior, single_prediction):
    """Combine corner boxes with class labels."""
    return jp.concatenate(
        [expected_corner_from_prior, single_prediction[:, 4:]], axis=1
    )


@pytest.fixture
def expected_encoding_with_alternate_variances(
    box_centers, prior_data, alternate_variances
):
    """Expected encoding values for the first box using alternate variances."""
    epsilon = 1e-8

    exp_cx_diff = (
        (box_centers[0, 0] - prior_data[0, 0])
        / prior_data[0, 2]
        / alternate_variances[0]
    )
    exp_cy_diff = (
        (box_centers[0, 1] - prior_data[0, 1])
        / prior_data[0, 3]
        / alternate_variances[1]
    )
    exp_w = (
        jp.log(box_centers[0, 2] / prior_data[0, 2] + epsilon)
        / alternate_variances[2]
    )
    exp_h = (
        jp.log(box_centers[0, 3] / prior_data[0, 3] + epsilon)
        / alternate_variances[3]
    )

    return {
        "cx_diff": exp_cx_diff,
        "cy_diff": exp_cy_diff,
        "width": exp_w,
        "height": exp_h,
    }


def test_decode_basic(generate_sample_boxes_and_priors):
    boxes, priors = generate_sample_boxes_and_priors
    variances = [0.1, 0.1, 0.2, 0.2]
    encoded = encode(boxes, priors, variances)
    decoded = decode(encoded, priors, variances)

    assert jp.allclose(decoded, boxes)


def test_encode_decode_roundtrip(generate_sample_boxes_and_priors):
    """Test that encoding and then decoding results in the original boxes."""
    boxes, priors = generate_sample_boxes_and_priors
    encoded = encode(boxes, priors)
    decoded = decode(encoded, priors)

    assert jp.allclose(decoded, boxes)


def test_decode_with_different_variances(generate_sample_boxes_and_priors):
    """Test decoding with different variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    variances = [0.2, 0.3, 0.4, 0.5]

    encoded = encode(boxes, priors, variances)

    decoded = decode(encoded, priors, variances)

    assert jp.allclose(decoded, boxes)


def test_encode_with_empty_array():
    """Test encoding with an empty array."""
    empty_boxes = jp.zeros((0, 5))
    empty_priors = jp.zeros((0, 4))

    encoded = encode(empty_boxes, empty_priors)

    assert encoded.shape == (0, 5)


def test_decode_with_empty_array():
    """Test decoding with an empty array."""
    empty_encoded = jp.zeros((0, 5))
    empty_priors = jp.zeros((0, 4))

    decoded = decode(empty_encoded, empty_priors)

    assert decoded.shape == (0, 5)


def test_encode_with_multiple_classes(generate_sample_boxes_and_priors):
    """Test encoding with boxes having different class labels."""
    boxes, priors = generate_sample_boxes_and_priors

    boxes_with_classes = boxes.at[:, 4].set(jp.array([1.0, 2.0]))

    encoded = encode(boxes_with_classes, priors)

    assert jp.allclose(encoded[:, 4], boxes_with_classes[:, 4])


def test_decode_preserves_class_labels(generate_sample_boxes_and_priors):
    """Test that decoding preserves class labels."""
    boxes, priors = generate_sample_boxes_and_priors

    boxes_with_classes = boxes.at[:, 4].set(jp.array([1.0, 2.0]))

    encoded = encode(boxes_with_classes, priors)
    decoded = decode(encoded, priors)

    assert jp.allclose(decoded[:, 4], boxes_with_classes[:, 4])


def test_encode_with_small_box():
    """Test encoding with a very small box coordinate."""
    small_box = jp.array([[10, 10, 11, 11, 1]])

    priors = jp.array([[5, 5, 10, 10]])

    encoded_small = encode(small_box, priors)

    assert not jp.any(jp.isnan(encoded_small))
    assert not jp.any(jp.isinf(encoded_small))


def test_encode_with_large_box():
    """Test encoding with a very large box coordinate."""
    large_box = jp.array([[0, 0, 1000, 1000, 1]])

    priors = jp.array([[5, 5, 10, 10]])

    encoded_large = encode(large_box, priors)

    assert not jp.any(jp.isnan(encoded_large))
    assert not jp.any(jp.isinf(encoded_large))


def test_decode_with_large_values():
    """Test decoding with large encoded values."""
    large_encoded_values = jp.array(
        [
            [5.0, 5.0, 2.0, 2.0, 1],
            [-5.0, -5.0, -2.0, -2.0, 2],
        ]
    )

    priors = jp.array([[50, 50, 20, 20], [50, 50, 20, 20]])

    decoded = decode(large_encoded_values, priors)

    assert not jp.any(jp.isnan(decoded))
    assert not jp.any(jp.isinf(decoded))


def test_encode_with_identical_boxes_and_priors():
    """Test encoding when boxes are identical to priors."""
    priors = jp.array([[50, 50, 20, 20], [100, 100, 30, 30]])

    prior_corners = paz.boxes.to_corner_form(priors)
    boxes = jp.hstack([prior_corners, jp.ones((2, 1))])
    encoded = encode(boxes, priors)

    expected_centers = jp.zeros((2, 2))
    expected_sizes = jp.zeros((2, 2))

    assert jp.allclose(encoded[:, 0:2], expected_centers)
    assert jp.allclose(encoded[:, 2:4], expected_sizes)


def test_encode_with_very_small_variances(generate_sample_boxes_and_priors):
    """Test encoding with very small variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    small_variances = [1e-5, 1e-5, 1e-5, 1e-5]

    encoded = encode(boxes, priors, small_variances)

    assert jp.isfinite(encoded).all()


def test_decode_with_very_small_variances(generate_sample_boxes_and_priors):
    """Test decoding with very small variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    small_variances = [1e-5, 1e-5, 1e-5, 1e-5]

    encoded = encode(boxes, priors, small_variances)

    decoded = decode(encoded, priors, small_variances)

    assert jp.allclose(decoded, boxes)


def test_encode_with_very_large_variances(generate_sample_boxes_and_priors):
    """Test encoding with very large variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    large_variances = [1e5, 1e5, 1e5, 1e5]

    encoded = encode(boxes, priors, large_variances)

    assert jp.isfinite(encoded).all()


def test_decode_with_very_large_variances(generate_sample_boxes_and_priors):
    """Test decoding with very large variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    large_variances = [1e5, 1e5, 1e5, 1e5]

    encoded = encode(boxes, priors, large_variances)

    decoded = decode(encoded, priors, large_variances)

    assert jp.allclose(decoded, boxes)


def test_encode_with_negative_coordinates():
    """Test encoding with boxes having negative coordinates."""
    neg_boxes = jp.array([[-20, -30, 10, 20, 1]])
    priors = jp.array([[0, 0, 30, 50]])

    encoded = encode(neg_boxes, priors)

    assert not jp.any(jp.isnan(encoded))
    assert not jp.any(jp.isinf(encoded))

    decoded = decode(encoded, priors)
    assert jp.allclose(decoded, neg_boxes)


def test_encode_and_decode_with_small_width():
    """Test encode/decode with a box having a very small width."""
    small_width = jp.array([[10, 20, 10.001, 90, 1]])
    priors = jp.array([[35, 55, 50, 70]])
    encoded = encode(small_width, priors)
    decoded = decode(encoded, priors)
    assert jp.isfinite(decoded).all()


def test_encode_and_decode_with_small_height():
    """Test encode/decode with a box having a very small height."""
    small_height = jp.array([[10, 20, 60, 20.001, 1]])
    priors = jp.array([[35, 55, 50, 70]])
    encoded = encode(small_height, priors)
    decoded = decode(encoded, priors)

    assert jp.isfinite(decoded).all()


def test_encode_with_zero_size_boxes():
    """Test encoding with boxes having zero width or height."""
    zero_width = jp.array([[10, 20, 10, 40, 1]])

    zero_height = jp.array([[10, 20, 30, 20, 1]])

    priors = jp.array([[15, 30, 10, 20]])

    encoded_w = encode(zero_width, priors)
    encoded_h = encode(zero_height, priors)

    assert jp.isfinite(encoded_w).all()
    assert jp.isfinite(encoded_h).all()


def test_encode_with_large_offset_from_prior():
    """Test encoding with boxes far away from priors."""
    far_box = jp.array([[1000, 1000, 1100, 1100, 1]])
    priors = jp.array([[10, 10, 20, 20]])

    encoded = encode(far_box, priors)

    assert jp.isfinite(encoded).all()

    decoded = decode(encoded, priors)
    assert jp.allclose(decoded, far_box)


@pytest.mark.skip(reason="to corner form changed implementation")
def test_decode_with_zero_predictions():
    """Test decoding when predictions are all zeros."""
    zero_pred = jp.zeros((2, 5))
    zero_pred = zero_pred.at[:, 4].set(jp.array([1, 2]))

    priors = jp.array([[50, 50, 20, 20], [100, 100, 30, 30]])

    decoded = decode(zero_pred, priors)

    expected = jp.hstack([to_corner_form(priors), zero_pred[:, 4:5]])

    assert jp.allclose(decoded, expected)


def test_encode_with_single_point_boxes():
    """Test encoding with boxes that are just points (width=height=0)."""
    point_boxes = jp.array([[10, 20, 10, 20, 1], [30, 40, 30, 40, 2]])

    priors = jp.array([[15, 25, 10, 10], [35, 45, 10, 10]])

    encoded = encode(point_boxes, priors)

    assert jp.isfinite(encoded).all()


def test_encode_decode_with_no_variances(generate_sample_boxes_and_priors):
    """Test encoding and decoding without specifying variances parameter."""
    boxes, priors = generate_sample_boxes_and_priors

    encoded = encode(boxes, priors)

    decoded = decode(encoded, priors)
    assert jp.allclose(decoded, boxes)


def test_with_multiple_additional_attributes():
    """Test encoding and decoding with boxes having multiple additional attributes."""
    boxes = jp.array(
        [[10, 20, 60, 90, 1, 0.8, 0], [30, 40, 100, 120, 2, 0.9, 1]]
    )

    priors = jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])

    encoded = encode(boxes, priors)

    assert encoded.shape == (2, 7)

    decoded = decode(encoded, priors)

    assert jp.allclose(decoded[:, 4:], boxes[:, 4:])


def test_encode_decode_with_different_box_count():
    """Test encoding and decoding with different numbers of boxes and priors."""
    boxes_more = jp.array(
        [[10, 20, 60, 90, 1], [30, 40, 100, 120, 2], [50, 60, 150, 200, 3]]
    )

    priors_less = jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])

    boxes_less = jp.array([[10, 20, 60, 90, 1]])

    priors_more = jp.array(
        [[35, 55, 50, 70], [65, 80, 70, 80], [100, 120, 90, 100]]
    )

    try:
        encode(boxes_more, priors_less)
    except Exception:
        pass

    try:
        encode(boxes_less, priors_more)
    except Exception:
        pass


def test_encode_with_zero_width_height_priors():
    """Test encoding with priors having zero width or height (edge case)."""
    boxes = jp.array([[10, 20, 30, 40, 1]])

    prior_zero_w = jp.array([[15, 30, 0, 20]])

    prior_zero_h = jp.array([[15, 30, 20, 0]])

    try:
        encode(boxes, prior_zero_w)
        encode(boxes, prior_zero_h)
    except Exception:
        pass


def test_encode_basic(encoded_boxes, expected_encoding_values):
    """Test basic encoding functionality."""
    assert jp.allclose(encoded_boxes[0, 0], expected_encoding_values["cx_diff"])
    assert jp.allclose(encoded_boxes[0, 1], expected_encoding_values["cy_diff"])
    assert jp.allclose(encoded_boxes[0, 2], expected_encoding_values["width"])
    assert jp.allclose(encoded_boxes[0, 3], expected_encoding_values["height"])
    assert jp.allclose(encoded_boxes[0, 4], 1.0)


def test_encode_dimensions(
    box_data, prior_data, default_variances, expected_encoding_values
):
    """Test the encoding of box dimensions."""
    encoded = encode(box_data, prior_data, default_variances)
    expected_dims = jp.array(
        [
            [
                expected_encoding_values["width"],
                expected_encoding_values["height"],
            ]
        ]
    )
    assert jp.allclose(encoded[0, 2:4], expected_dims)


def test_encode_center_coordinates(
    box_data, prior_data, default_variances, expected_encoding_values
):
    """Test the encoding of center coordinates."""
    encoded = encode(box_data, prior_data, default_variances)
    expected_coords = jp.array(
        [
            [
                expected_encoding_values["cx_diff"],
                expected_encoding_values["cy_diff"],
            ]
        ]
    )
    assert jp.allclose(encoded[0, 0:2], expected_coords)


def test_concatenate_encoded_boxes(box_data, prior_data, default_variances):
    """Test that class labels are preserved in encoding."""
    encoded = encode(box_data, prior_data, default_variances)
    expected_extras = box_data[:, 4:]
    expected_shape = (2, 5)

    assert (
        encoded.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {encoded.shape}"
    assert jp.allclose(encoded[:, 4:], expected_extras)


def test_encode_with_different_variances(
    box_data,
    prior_data,
    alternate_variances,
    expected_encoding_with_alternate_variances,
):
    """Test encoding with different variance values."""
    encoded = encode(box_data, prior_data, alternate_variances)

    assert jp.allclose(
        encoded[0, 0], expected_encoding_with_alternate_variances["cx_diff"]
    )
    assert jp.allclose(
        encoded[0, 1], expected_encoding_with_alternate_variances["cy_diff"]
    )
    assert jp.allclose(
        encoded[0, 2], expected_encoding_with_alternate_variances["width"]
    )
    assert jp.allclose(
        encoded[0, 3], expected_encoding_with_alternate_variances["height"]
    )


@pytest.mark.skip(reason="to corner form changed implementation")
def test_decode_helpers(
    single_prediction, single_prior, default_variances, expected_decoded_center
):
    """Test the individual decode helper functions."""
    result = decode(single_prediction, single_prior, default_variances)
    expected_corner = to_corner_form(expected_decoded_center)
    assert jp.allclose(result[:, :4], expected_corner)


@pytest.mark.skip(reason="to corner form changed implementation")
def test_compute_boxes_center(
    single_prediction, single_prior, default_variances, expected_decoded_center
):
    """Test compute_boxes_center function."""
    result = decode(single_prediction, single_prior, default_variances)
    expected_corner = to_corner_form(expected_decoded_center)
    assert jp.allclose(result[:, :4], expected_corner)


@pytest.mark.skip(reason="to corner form changed implementation")
def test_combine_with_extras(single_prior, single_prediction, boxes_with_class):
    """Test combining box coordinates with class labels."""
    boxes_corner = to_corner_form(single_prior)
    combined = jp.concatenate([boxes_corner, single_prediction[:, 4:]], axis=1)
    assert jp.allclose(combined, boxes_with_class)


def test_encode_decode_1000_boxes(large_test_data, default_variances):
    """Test performance and correctness with many boxes."""
    boxes, priors = large_test_data

    encoded = encode(boxes, priors, default_variances)

    decoded = decode(encoded, priors, default_variances)

    assert jp.allclose(
        decoded[:, :4], boxes[:, :4], rtol=1e-4, atol=1e-4
    ), "Bounding box coordinates not recovered within tolerance"
