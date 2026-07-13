import jax
import jax.numpy as jp

from paz.backend import features


def compute_keypoint_heatmap(logits):
    scores = jax.nn.softmax(logits, axis=-1)[..., :64]
    height, width = scores.shape[0], scores.shape[1]
    scores = jp.reshape(scores, (height, width, 8, 8))
    scores = jp.transpose(scores, (0, 2, 1, 3))
    return jp.reshape(scores, (height * 8, width * 8))


def compute_dense_scores(heatmap, heat, threshold, height, width):
    grid = features.build_pixel_grid(heatmap.shape[0], heatmap.shape[1])
    dense = heatmap[..., None]
    sharp = features.sample_features(dense, grid, height, width, "nearest")
    soft = features.sample_features(heat, grid, height, width, "bilinear")
    keep = features.find_local_maxima(heatmap, 5) & (heatmap > threshold)
    scores = jp.where(keep.reshape(-1), (sharp * soft)[:, 0], -1.0)
    return grid, scores
