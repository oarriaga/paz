
def validate_num_points(min_points):
    num_points = len(points1)
    if num_points < min_samples:
        print("Warning: Not enough points provided for RANSAC.")



def ransac(key, points1, points2, min_samples, threshold, max_trials):
