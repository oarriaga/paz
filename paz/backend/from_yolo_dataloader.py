import os
import cv2
import numpy as np
from tqdm import tqdm

# Define a tuple of valid image file extensions (lowercase)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def validate_directories(images_directory: str, labels_directory: str):
    """Validates that the provided directories exist."""
    if not os.path.isdir(images_directory):
        raise FileNotFoundError(
            f"Images directory does not exist: {images_directory}"
        )
    if not os.path.isdir(labels_directory):
        raise FileNotFoundError(
            f"Labels directory does not exist: {labels_directory}"
        )


def validate_file_correspondence(image_files: list, label_files: list):
    """
    Validates that each image file has a corresponding label file
    and vice versa.

    Args:
        image_files (list of str): List of paths to image files.
        label_files (list of str): List of paths to label files.

    Raises:
        ValueError: If there are image files without corresponding label files
                    or label files without corresponding image files.
                    The error message will list the missing files.
    """

    # Build a set of base names (without extension) for image files
    image_bases = {
        os.path.splitext(image_file)[0] for image_file in image_files
    }
    # Build a set of base names for label files (assuming they are .txt)
    label_bases = {
        os.path.splitext(label_file)[0] for label_file in label_files
    }

    missing_labels = []
    for base in image_bases:
        if base not in label_bases:
            missing_labels.append(base + ".txt")
    if missing_labels:
        raise ValueError(
            "The following image files have no corresponding label files: "
            + str(missing_labels)
        )

    missing_images = []
    for base in label_bases:
        if base not in image_bases:
            missing_images.append(base)
    if missing_images:
        raise ValueError(
            "The following label files have no corresponding image files: "
            + str([img_base + ".[ext]" for img_base in missing_images])
        )


def get_image_files(images_directory: str):
    """Returns a list of image files from the images_directory
    matching the valid extensions.
    """
    image_files = [
        f
        for f in os.listdir(images_directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not image_files:
        raise FileNotFoundError(
            f"No image files found in the specified directory: {images_directory}"
        )
    return image_files


def get_label_files(labels_directory: str):
    """Returns a list of label files with a .txt extension from the labels_directory."""
    label_files = [
        f for f in os.listdir(labels_directory) if f.endswith(".txt")
    ]
    if not label_files:
        raise FileNotFoundError(
            f"No label files found in the specified directory: {labels_directory}"
        )
    return label_files


def get_image_size(image_path: str):
    """
    Returns the size of the image at the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (width, height) of the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    height, width, _ = image.shape
    return width, height


def load_label(label_path: str):
    """
    Loads labels from a file.

    Args:
        label_path (str): Path to the label file.

    Returns:
        np.ndarray: Array of labels, where each label is a list of floats.
    """
    labels = []
    with open(label_path, "r") as file:
        for line in file:
            # Convert each line to a list of floats
            labels.append([float(x) for x in line.strip().split()])
    return np.array(labels)


def process_labels(
    labels: np.ndarray, image_size: tuple, normalize: bool = True
):
    """
    Processes labels by converting coordinates to absolute pixel coordinates.

    Args:
        labels (np.ndarray): Array of labels where each label is in the format
                             [class_id, x_center, y_center, width, height].
        image_size (tuple): The (width, height) of the image.
        normalize (bool): If True, assumes the coordinates are normalized
                        (0 to 1 range). If False, assumes absolute coordinates.

    Returns:
        np.ndarray: Array of processed labels in the format
                    [x_min, y_min, x_max, y_max, class_id].
    """
    processed_labels = []
    width, height = image_size  # assuming image_size is (width, height)
    for label in labels:
        class_id, x_center, y_center, box_width, box_height = label
        if normalize:
            # When coordinates are normalized, convert to absolute coordinates later
            x_min = x_center - box_width / 2
            x_max = x_center + box_width / 2
            y_min = y_center - box_height / 2
            y_max = y_center + box_height / 2
        else:
            # When coordinates are already absolute (scaled by image size)
            x_min = int((x_center - box_width / 2) * width)
            y_min = int((y_center - box_height / 2) * height)
            x_max = int((x_center + box_width / 2) * width)
            y_max = int((y_center + box_height / 2) * height)

        processed_labels.append([x_min, y_min, x_max, y_max, int(class_id)])
    return np.array(processed_labels)


def get_data_PAZ_formate(
    images_directory: str, labels_directory: str, normalize: bool = True
):
    """
    Loads and processes image and label data
    for object detection in PAZ format.

    Args:
        images_directory (str): Path to the directory containing image files.
        labels_directory (str): Path to the directory containing label files.
        normalize (bool): Whether to normalize label coordinates.

    Returns:
        list: A list of dictionaries, each containing:
            - 'image' (str): Path to the image file.
            - 'boxes' (np.ndarray): Processed label data for the image.
    """
    data = []
    validate_directories(images_directory, labels_directory)
    image_files = get_image_files(images_directory)
    label_files = get_label_files(labels_directory)
    validate_file_correspondence(image_files, label_files)
    for file in tqdm(image_files, desc="Processing Data"):
        image_path = os.path.join(images_directory, file)
        image_size = get_image_size(image_path)
        # Get the base name (without extension) and build the label file name
        base_name = os.path.splitext(file)[0]
        label_path = os.path.join(labels_directory, base_name + ".txt")

        labels = load_label(label_path)
        box_data = process_labels(labels, image_size, normalize=normalize)

        data.append({"image": image_path, "boxes": box_data})
    return data


if __name__ == "__main__":
    images_directory_full_path = r"./Data/processed/yolo_dataset/images/train"
    labels_directory_full_path = r"./Data/processed/yolo_dataset/labels/train"

    data = get_data_PAZ_formate(
        images_directory_full_path, labels_directory_full_path, normalize=False
    )
    print(f"Number of data entries: {len(data)}")
    print(f"First data entry: {data[0]}")

    print("First data entry image path: ", data[0]["image"])
    print("First data entry box data: ", data[0]["boxes"])
