import jax.numpy as jp
import jax


def calculate_distances(training_features, query_point):
    differences = training_features - query_point
    squared_differences = jp.square(differences)
    sum_squared_differences = jp.sum(squared_differences, axis=1)
    return jp.sqrt(sum_squared_differences)


def get_nearest_neighbors_labels(distances, training_labels, number_of_neighbors):  # fmt: skip
    negative_distances = -distances
    top_k_values, top_k_indices = jax.lax.top_k(negative_distances, number_of_neighbors)  # fmt: skip
    nearest_labels = training_labels[top_k_indices]
    return nearest_labels


def calculate_mode(nearest_labels, number_of_classes):
    counts = jp.bincount(nearest_labels, length=number_of_classes)
    most_frequent_label = jp.argmax(counts)
    return most_frequent_label


def predict_single_instance(training_features, training_labels, query_point, number_of_neighbors, number_of_classes):  # fmt: skip
    distances = calculate_distances(training_features, query_point)
    nearest_labels = get_nearest_neighbors_labels(distances, training_labels, number_of_neighbors)  # fmt: skip
    predicted_label = calculate_mode(nearest_labels, number_of_classes)
    return predicted_label


def predict(training_features, training_labels, query_features, number_of_neighbors, number_of_classes):  # fmt: skip
    partial_predict = lambda query_point: predict_single_instance( training_features, training_labels, query_point, number_of_neighbors, number_of_classes)  # fmt: skip
    vectorized_predict = jax.vmap(partial_predict)
    return vectorized_predict(query_features)
