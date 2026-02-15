import jax
import jax.numpy as jp


def estimate_fundamental_matrix_ransac_jax(
    key,
    points1,
    points2,
    min_samples=8,
    residual_threshold=0.5,
    max_trials=1000,
):
    """
    Estimate the fundamental matrix between two sets of corresponding
    points using RANSAC
    """
    n_points = points1.shape[0]
    if n_points < min_samples:
        print("Warning: Not enough points provided for RANSAC.")
        return (jp.full((3, 3), jp.nan), jp.zeros(n_points, dtype=bool))

    # Initial state for the loop
    # Initialize best_F with NaNs to indicate no model found yet
    # Initialize best_inliers with all False
    initial_state = (
        key,  # JAX PRNG key
        jp.full((3, 3), jp.nan),  # best_F
        jp.zeros(n_points, dtype=bool),  # best_inliers
        jp.int32(0),  # best_num_inliers
        jp.inf,  # best_distance_sum (using squared distance)
    )

    def ransac_body(i, state):
        """Body function for one RANSAC iteration."""
        (
            key,
            current_best_F,
            current_best_inliers,
            current_best_num_inliers,
            current_best_distance_sum,
        ) = state
        subkey, key = jax.random.split(key)  # Split key for this iteration

        # 1. Select random sample
        indices = jax.random.choice(
            subkey, n_points, shape=(min_samples,), replace=False
        )
        sample_points1 = points1[indices]
        sample_points2 = points2[indices]

        # 2. Compute model (Fundamental Matrix)
        # Use lax.cond to handle potential failures in compute_fundamental_matrix_jax
        # This simple example assumes it always returns a valid matrix or NaNs
        F = compute_fundamental_matrix_jax(sample_points1, sample_points2)

        # Check if F computation was valid (e.g., not all NaNs)
        # If compute_fundamental_matrix_jax handles failures by returning NaNs:
        is_F_valid = ~jp.any(jp.isnan(F))

        # 3. Evaluate model on all points (if F is valid)
        # Use jp.where or calculation that handles potential NaNs in F gracefully
        # or skip evaluation if F is invalid.
        # Here, we calculate even if invalid, but use is_F_valid later.
        # Note: Ensure compute_sampson_distance handles potential NaN F if needed.
        #       The dummy version might produce NaNs if F has NaNs.
        distance_sq = compute_sampson_distance(
            F, points1, points2
        )  # Often Sampson^2 is used
        distance = jp.sqrt(
            jp.abs(distance_sq)
        )  # Get absolute distance if needed

        inliers = (
            distance < residual_threshold
        ) & is_F_valid  # Only valid F can have inliers
        num_inliers = jp.count_nonzero(inliers)
        # Using squared distance for sum to avoid sqrt, ensure consistency
        # Only sum distances for valid points if necessary (or mask invalid distances)
        # Here we sum all, but the model comparison logic handles num_inliers.
        distance_sum_sq = jp.dot(
            distance_sq, distance_sq
        )  # Sum of squares of distances

        # 4. Update best model if current is better
        # Criteria: More inliers OR same number of inliers but smaller error sum
        is_better = (num_inliers > current_best_num_inliers) | (
            (num_inliers == current_best_num_inliers)
            & (distance_sum_sq < current_best_distance_sum)
        )

        # Select outputs based on whether the current model is better
        best_F = jp.where(is_better, F, current_best_F)
        best_inliers = jp.where(is_better, inliers, current_best_inliers)
        best_num_inliers = jp.where(
            is_better, num_inliers, current_best_num_inliers
        )
        best_distance_sum = jp.where(
            is_better, distance_sum_sq, current_best_distance_sum
        )

        # Also ensure that if F was invalid, we don't update to it unless it's the very first step
        # and the initial state was NaN. A more robust way is needed if compute_F can fail often.
        # Simplified: The is_better logic combined with initial state should handle this reasonably.
        # If F is invalid, num_inliers will be 0, unlikely to be chosen unless current_best_num_inliers is also 0.

        return (key, best_F, best_inliers, best_num_inliers, best_distance_sum)

    final_state = jax.lax.fori_loop(0, max_trials, ransac_body, initial_state)
    _, best_F, best_inliers, best_num_inliers, _ = final_state
    return best_F, best_inliers
