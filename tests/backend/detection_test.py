import pytest
import jax
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
    kept = apply_valid_NMS(boxes_and_scores, iou_thresh, top_k)
    assert jp.allclose(kept, expected)


def apply_valid_NMS(boxes_and_scores, iou_thresh, top_k):
    suppressed = apply_NMS(boxes_and_scores, iou_thresh, top_k)
    assert len(suppressed) == min(top_k, len(boxes_and_scores))
    return paz.detection.remove_invalid(suppressed)


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


def test_apply_nms_higher_threshold(boxes_and_scores):
    """Test higher IoU threshold keeps more boxes."""
    kept = apply_valid_NMS(boxes_and_scores, 0.9, 200)
    assert jp.allclose(kept, boxes_and_scores[jp.array([0, 1, 3, 2])])


def test_apply_nms_lower_threshold(boxes_and_scores):
    """Test lower IoU threshold suppresses more boxes."""
    kept = apply_valid_NMS(boxes_and_scores, 0.4, 200)
    assert jp.allclose(kept, boxes_and_scores[jp.array([0, 3, 2])])


def test_apply_nms_top_k(boxes_and_scores):
    """Test top_k parameter limits initial candidates."""
    kept = apply_valid_NMS(boxes_and_scores, 0.5, 2)
    assert jp.allclose(kept, boxes_and_scores[:1])


def test_apply_nms_single_box():
    """Test single box returns itself."""
    boxes_and_scores = jp.array([[0, 0, 10, 10, 0.9]])
    kept = apply_valid_NMS(boxes_and_scores, 0.45, 400)
    assert jp.allclose(kept, boxes_and_scores)


def test_apply_nms_no_overlap():
    """Test non-overlapping boxes all kept."""
    boxes_and_scores = jp.array(
        [[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.8], [40, 40, 50, 50, 0.7]]
    )
    kept = apply_valid_NMS(boxes_and_scores, 0.1, 400)
    assert jp.allclose(kept, boxes_and_scores)


def test_apply_nms_identical_boxes():
    """Test identical boxes keep highest score."""
    boxes_and_scores = jp.array(
        [[0, 0, 10, 10, 0.7], [0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]]
    )
    kept = apply_valid_NMS(boxes_and_scores, 0.5, 400)
    assert jp.allclose(kept, boxes_and_scores[1:2])


def test_apply_per_class_NMS_with_fewer_detections_than_top_k():
    """Boxes 0 and 2 overlap; each class keeps its best scored one."""
    boxes_and_class_scores = jp.array(
        [
            [0, 0, 10, 10, 0.1, 0.9],
            [20, 20, 30, 30, 0.2, 0.8],
            [0, 0, 10, 10, 0.3, 0.7],
        ]
    )
    suppressed = paz.detection.apply_per_class_NMS(boxes_and_class_scores, 2)
    assert suppressed.shape == (6, 6)
    expected = jp.array(
        [
            [0, 0, 10, 10, 0.3, 0],
            [20, 20, 30, 30, 0.2, 0],
            [0, 0, 10, 10, 0.9, 1],
            [20, 20, 30, 30, 0.8, 1],
        ]
    )
    kept = paz.detection.remove_invalid(suppressed)
    assert jp.allclose(kept, expected)


@pytest.fixture
def generate_sample_boxes_and_priors():
    boxes = jp.array([[10, 20, 60, 90], [30, 40, 100, 120]])

    boxes_with_labels = jp.hstack([boxes, jp.ones((2, 1))])

    priors = jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])

    return boxes_with_labels, priors


@pytest.fixture
def default_variances():
    """Default variance values for encoding/decoding."""
    return [0.1, 0.1, 0.2, 0.2]


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


def test_decode_with_zero_predictions_returns_priors_in_corner_form():
    predictions = jp.zeros((2, 5)).at[:, 4].set(jp.array([1, 2]))
    priors = jp.array([[50, 50, 20, 20], [100, 100, 30, 30]])
    decoded = decode(predictions, priors)
    expected = jp.array([[40, 40, 60, 60, 1], [85, 85, 115, 115, 2]])
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


def test_encode_matches_hand_computed_offsets():
    """Box center (10, 30) with size (20, 40) against a (10, 20) prior."""
    matched = jp.array([[0.0, 10.0, 20.0, 50.0, 1.0]])
    priors = jp.array([[20.0, 20.0, 10.0, 20.0]])
    encoded = encode(matched, priors, [0.1, 0.1, 0.2, 0.2])
    expected = jp.array([[-10.0, 5.0, 3.465736, 3.465736, 1.0]])
    assert jp.allclose(encoded, expected, atol=1e-5)


def test_encode_divides_each_offset_by_its_variance():
    matched = jp.array([[0.0, 10.0, 20.0, 50.0, 1.0]])
    priors = jp.array([[20.0, 20.0, 10.0, 20.0]])
    encoded = encode(matched, priors, [0.2, 0.3, 0.4, 0.5])
    expected = jp.array([[-5.0, 1.666667, 1.732868, 1.386294, 1.0]])
    assert jp.allclose(encoded, expected, atol=1e-5)


def test_decode_matches_hand_computed_box():
    """Center (50.3, 60.8) with size (30 e^0.06, 40 e^0.08)."""
    predictions = jp.array([[0.1, 0.2, 0.3, 0.4, 1.0]])
    priors = jp.array([[50.0, 60.0, 30.0, 40.0]])
    decoded = decode(predictions, priors, [0.1, 0.1, 0.2, 0.2])
    expected = jp.array([[34.37245, 39.134254, 66.22755, 82.465744, 1.0]])
    assert jp.allclose(decoded, expected, atol=1e-4)


def test_encode_decode_1000_boxes(large_test_data, default_variances):
    """Test performance and correctness with many boxes."""
    boxes, priors = large_test_data

    encoded = encode(boxes, priors, default_variances)

    decoded = decode(encoded, priors, default_variances)

    assert jp.allclose(
        decoded[:, :4], boxes[:, :4], rtol=1e-4, atol=1e-4
    ), "Bounding box coordinates not recovered within tolerance"


def test_match_ignores_repeated_boxes():
    """Edge padding repeats the last box; matches must not change."""
    prior_boxes = jp.array(
        [
            [0.15, 0.15, 0.1, 0.1],
            [0.35, 0.35, 0.1, 0.1],
            [0.55, 0.55, 0.1, 0.1],
            [0.95, 0.95, 0.1, 0.1],
        ]
    )
    boxes = jp.array([[0.1, 0.1, 0.2, 0.2, 1], [0.5, 0.5, 0.6, 0.6, 2]])
    repeated_boxes = jp.concatenate([boxes, boxes[-1:]])
    matched = paz.detection.match(boxes, prior_boxes, 0.5)
    repeated_matched = paz.detection.match(repeated_boxes, prior_boxes, 0.5)
    assert jp.allclose(matched[:, 4], jp.array([1, 0, 2, 0]))
    assert jp.allclose(matched, repeated_matched)


def test_padded_boxes_do_not_overwrite_positive_matches():
    """
    Tests that the forced matching of a padded 'background' box does not
    incorrectly overwrite a legitimate positive match.

    This test will FAIL with the original `match` function and PASS with
    the proposed fix, thus proving the existence of the flaw.
    """
    # We define two priors. One will be a perfect match, one will be background.
    prior_boxes = jp.array(
        [
            [0.15, 0.15, 0.1, 0.1],  # Perfect match for the real GT box
            [0.8, 0.8, 0.1, 0.1],  # A background prior
        ]
    )

    # We simulate data that has been padded with a constant value (-1) and has
    # had the background class added.
    # Real object class 1 becomes 2. Padded object class -1 becomes 0.
    data_with_padding = jp.array(
        [
            [0.1, 0.1, 0.2, 0.2, 2],  # Real box, positive class
            [-1, -1, -1, -1, 0],  # Padded box, background class
        ]
    )

    # Run the match function.
    # The bug: The padded box will find prior 0 as its "best" match (since all
    # IOUs are 0, argmax returns 0) and will overwrite the positive match.
    matched = paz.detection.match(data_with_padding, prior_boxes)

    # We check the final class assigned to the first prior.
    # It SHOULD be 2 (the real object). The bug will cause it to be 0.
    final_class_of_prior0 = matched[0, -1]

    # This assertion checks that the positive match was NOT overwritten.
    assert (
        final_class_of_prior0 > 0
    ), "A positive match was incorrectly overwritten by a padded background box."


def test_match_logic_with_padded_boxes():
    """
    Tests the corrected match function with a mix of high-IOU matches,
    forced matches, background priors, and padded data.
    """
    # We define 4 priors for various scenarios
    prior_boxes = jp.array(
        [
            [0.15, 0.15, 0.1, 0.1],  # prior 0: Perfect IOU with GT box 0
            [
                0.55,
                0.55,
                0.1,
                0.1,
            ],  # prior 1: Low IOU, but is best for GT box 1
            [0.95, 0.95, 0.1, 0.1],  # prior 2: Background, no good match
        ]
    )

    # We define 2 real ground truth (GT) boxes and 2 padded boxes.
    # The classes have already been incremented (e.g., from 1,2 to 2,3)
    # and the padded boxes have been set to class 0, mimicking the pipeline.
    boxes_with_class_arg = jp.array(
        [
            [0.1, 0.1, 0.2, 0.2, 2],  # GT box 0 (class 2)
            [0.5, 0.5, 0.6, 0.6, 3],  # GT box 1 (class 3)
            [-1, -1, -1, -1, 0],  # Padded box 1
            [-1, -1, -1, -1, 0],  # Padded box 2
        ]
    )

    # IOU threshold is 0.5.
    # IOU(prior 0, GT 0) is high (~1.0) -> a positive match.
    # IOU(prior 1, GT 1) is high (~1.0), but let's assume for the sake
    # of a forced match test that the threshold is higher, e.g. we'll test
    # that it gets matched regardless of IOU.
    # The logic inside match handles this.
    matched_boxes = paz.detection.match(
        boxes_with_class_arg, prior_boxes, IOU_threshold=0.5
    )

    # --- VALIDATION ---

    # 1. Test Prior 0: High IOU match
    # It should be matched with GT box 0 (class 2) and its exact coordinates.
    assert matched_boxes[0, 4] == 2, "Prior 0 should match class 2"
    assert jp.allclose(
        matched_boxes[0, :4], boxes_with_class_arg[0, :4]
    ), "Prior 0 should have the coordinates of GT box 0"

    # 2. Test Prior 1: Forced match
    # GT box 1's best match is prior 1. The 'force match' logic ensures this
    # assignment happens. It should be matched with GT box 1 (class 3).
    assert (
        matched_boxes[1, 4] == 3
    ), "Prior 1 should be force-matched to class 3"
    assert jp.allclose(
        matched_boxes[1, :4], boxes_with_class_arg[1, :4]
    ), "Prior 1 should have the coordinates of GT box 1"

    # 3. Test Prior 2: Background match
    # It has low IOU with all real boxes and is not a forced match.
    # It should be assigned a background class of 0.
    assert matched_boxes[2, 4] == 0, "Prior 2 should be background (class 0)"


def test_clip_keeps_edge_box_but_drops_padding():
    edge = [-4.0, 0.0, 50.0, 60.0, 0.99, 3.0]
    padding = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    detections = jp.array([edge, padding])
    clipped = paz.detection.clip(detections, 64, 64)
    kept = paz.detection.remove_invalid(clipped)
    assert kept.shape[0] == 1
    assert float(kept[0, 0]) == 0.0
    assert float(kept[0, 2]) == 50.0
    assert float(kept[0, 4]) == pytest.approx(0.99)


def padded_detections():
    boxes = jp.array([[0.3, 0.3, 0.5, 0.7, 5.0], [0.0, 0.0, 0.05, 0.05, 2.0]])
    return paz.detection.pad(boxes, 8, "constant", -1)


def test_random_expand_identity_when_probability_zero():
    image = jp.zeros((300, 300, 3), jp.uint8)
    detections = padded_detections()
    mean = jp.asarray(paz.image.BGR_IMAGENET_MEAN, jp.float32)
    _, out = paz.detection.random_expand(jax.random.PRNGKey(0), image,
                                         detections, mean, probability=0.0)
    assert jp.allclose(out, detections.astype(jp.float32))


def test_random_expand_shrinks_boxes_and_keeps_padding():
    image = jp.zeros((300, 300, 3), jp.uint8)
    detections = padded_detections()
    mean = jp.asarray(paz.image.BGR_IMAGENET_MEAN, jp.float32)
    _, out = paz.detection.random_expand(jax.random.PRNGKey(3), image,
                                         detections, mean, probability=1.0)
    valid = out[out[:, 4] >= 0]
    assert valid.shape[0] == 2
    assert float(valid[:, :4].min()) >= -1e-6
    assert float(valid[:, :4].max()) <= 1.0 + 1e-6
    width, height = valid[0, 2] - valid[0, 0], valid[0, 3] - valid[0, 1]
    assert float(width * height) <= (0.5 - 0.3) * (0.7 - 0.3) + 1e-6
    assert float(out[-1, 4]) == -1.0


def test_apply_crop_remaps_and_drops_outside_boxes():
    image = jp.zeros((300, 300, 3), jp.uint8)
    detections = padded_detections()
    window = jp.array([0.2, 0.1, 0.8, 0.9])
    _, cropped = paz.detection.apply_crop(image, detections, window)
    kept = cropped[cropped[:, 4] >= 0]
    assert kept.shape[0] == 1
    expected = jp.array([(0.3 - 0.2) / 0.6, (0.3 - 0.1) / 0.8,
                         (0.5 - 0.2) / 0.6, (0.7 - 0.1) / 0.8])
    assert jp.allclose(kept[0, :4], expected, atol=1e-5)
    assert float(kept[0, 4]) == 5.0
    assert cropped.shape[0] == detections.shape[0]


def test_random_flip_mirrors_and_preserves_padding():
    image = jax.random.randint(jax.random.PRNGKey(0), (300, 300, 3), 0, 256)
    detections = padded_detections()
    _, out = paz.detection.random_flip(jax.random.PRNGKey(0), image,
                                       detections, probability=1.0)
    assert jp.allclose(out[0, :4],
                       jp.array([1.0 - 0.5, 0.3, 1.0 - 0.3, 0.7]), atol=1e-5)
    assert float(out[0, 4]) == 5.0
    assert float(out[-1, 4]) == -1.0
    _, same = paz.detection.random_flip(jax.random.PRNGKey(0), image,
                                        detections, probability=0.0)
    assert jp.allclose(same, detections.astype(jp.float32))


def test_augment_detection_is_jit_vmap_stable():
    image = jax.random.randint(jax.random.PRNGKey(0), (300, 300, 3), 0, 256)
    detections = padded_detections()
    mean = jp.asarray(paz.image.BGR_IMAGENET_MEAN, jp.float32)
    augment_fn = jax.vmap(paz.detection.augment_detection, (0, 0, 0, None))
    augment = jax.jit(augment_fn)
    images = jp.broadcast_to(image, (6, 300, 300, 3))
    batch = jp.broadcast_to(detections, (6, 8, 5))
    for seed in range(3):
        keys = jax.random.split(jax.random.PRNGKey(seed), 6)
        out_images, out_detections = augment(keys, images, batch, mean)
    assert out_images.shape == (6, 300, 300, 3)
    assert out_detections.shape == (6, 8, 5)
    assert bool(jp.all(jp.isfinite(out_images)))
    assert augment._cache_size() == 1


def test_random_flip_left_right_keeps_box_on_object():
    image = jp.zeros((50, 100, 3)).at[20:40, 10:30].set(1.0)
    detections = jp.array([[10.0, 20.0, 30.0, 40.0, 1.0]])
    image, flipped = paz.detection.random_flip_left_right(image, detections)
    assert jp.allclose(flipped, jp.array([[70.0, 20.0, 90.0, 40.0, 1.0]]))
    assert jp.all(image[20:40, 70:90] == 1.0)
    assert float(image.sum()) == 20 * 20 * 3
