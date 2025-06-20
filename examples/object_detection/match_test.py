# numpy (test_low_iou_match_is_background)
# numpy (test_forced_match_tie_break_by_iou)
# numpy (test_padded_box_does_not_interfere_with_good_match)

# match_1 (test_low_iou_match_is_background)

# match (test_low_iou_match_is_background)
# match (test_forced_match_tie_break_by_iou)

import pytest

use_numpy = False
# use_numpy = True

if use_numpy:
    # import jax.numpy as jp
    import numpy as jp
    from paz.backend.detection import match_np as match
else:
    import jax.numpy as jp

    # from paz.backend.detection import match_1 as match

    from paz.backend.detection import match


def test_high_iou_match_is_assigned():
    """Tests that a prior with a clear high-IOU match gets assigned correctly."""
    prior_boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])  # A single prior
    boxes_with_class_arg = jp.array(
        [[0.5, 0.5, 1.0, 1.0, 10]]  # A single GT box that overlaps perfectly
    )
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    assert matched_boxes[0, 4] == 10


def test_low_iou_match_is_background():
    """Tests that a prior with a low-IOU match is labeled as background."""
    prior_boxes = jp.array([[0.0, 0.0, 0.1, 0.1]])
    boxes_with_class_arg = jp.array([[0.9, 0.9, 0.1, 0.1, 10]])
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    assert matched_boxes[0, 4] == 0.0


def test_forced_match_is_positive_below_threshold():
    """Tests that a forced match is positive even if its IOU is below the threshold."""
    prior_boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    # This GT box has a low IOU, but it's the only one, so it must be force-matched.
    boxes_with_class_arg = jp.array([[0.0, 0.0, 0.1, 0.1, 99]])
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    assert matched_boxes[0, 4] == 99


def test_padded_box_is_ignored_in_forced_match():
    """Tests that a padded GT box does not create a positive forced match."""
    prior_boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    # A padded box should not be force-matched to the prior.
    boxes_with_class_arg = jp.array([[-1.0, -1.0, -1.0, -1.0, 0]])
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    assert matched_boxes[0, 4] == 0.0


def test_forced_match_tie_break_by_iou():
    """
    Tests that if two GT boxes are force-matched to the same prior,
    the one with the higher IOU wins the assignment.
    """
    prior_boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])  # Single prior
    boxes_with_class_arg = jp.array(
        [
            [0.4, 0.4, 0.6, 0.6, 10],  # GT A, class 10, higher IOU
            [0.0, 0.0, 0.2, 0.2, 99],  # GT B, class 99, lower IOU
        ]
    )
    # Both GT boxes' best (and only) prior is the one at index 0.
    # The logic must assign the prior to GT A because it has a higher IOU.
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.1)
    assert matched_boxes[0, 4] == 10


def test_forced_match_overwrites_threshold_match():
    """
    Tests that a forced match for a GT box correctly overwrites a prior's
    initial match that was only based on the IOU threshold.
    """
    prior_boxes = jp.array(
        [
            [0.15, 0.15, 0.1, 0.1],  # Prior A
            [0.45, 0.45, 0.2, 0.2],  # Prior B
        ]
    )
    boxes_with_class_arg = jp.array(
        [
            [0.4, 0.4, 0.5, 0.5, 10],  # GT A, high IOU with Prior B
            [0.1, 0.1, 0.2, 0.2, 99],  # GT B, best prior is Prior A
        ]
    )
    # - Prior B's best IOU match is GT A.
    # - GT B's best prior for a forced match is Prior A.
    # The logic must ensure GT B gets matched with Prior A, even though
    # Prior A had a decent match with nothing.
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    assert matched_boxes[0, 4] == 99


def test_correct_coordinates_are_assigned():
    """
    Tests that the coordinates of the assigned box are correct, not just the class.
    """
    prior_boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    boxes_with_class_arg = jp.array([[0.1, 0.2, 0.3, 0.4, 10]])
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.1)
    expected_coords = jp.array([0.1, 0.2, 0.3, 0.4])
    assert jp.allclose(matched_boxes[0, :4], expected_coords)


def test_single_padded_box_is_ignored():
    """Tests that a single padded GT box does not create a positive match."""
    prior_boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    # A padded box (class 0 or -1, with negative coords) should be ignored.
    boxes_with_class_arg = jp.array([[-1.0, -1.0, -1.0, -1.0, 0]])
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    assert matched_boxes[0, 4] == 0.0


def test_padded_box_does_not_interfere_with_good_match():
    """Tests that a padded box doesn't interfere with a clear high-IOU match."""
    prior_boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    boxes_with_class_arg = jp.array(
        [
            [0.5, 0.5, 1.0, 1.0, 10],  # Perfect match
            [-1.0, -1.0, -1.0, -1.0, 0],  # Padded box
        ]
    )
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    assert matched_boxes[0, 4] == 10


def test_all_padded_boxes_results_in_all_background():
    """Tests that if all GT boxes are padded, all priors are background."""
    prior_boxes = jp.array(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
        ]
    )
    boxes_with_class_arg = jp.array(
        [
            [-1.0, -1.0, -1.0, -1.0, 0],
            [-1.0, -1.0, -1.0, -1.0, 0],
        ]
    )
    matched_boxes = match(boxes_with_class_arg, prior_boxes, IOU_threshold=0.5)
    # All priors should be assigned class 0.
    assert jp.all(matched_boxes[:, 4] == 0.0)
