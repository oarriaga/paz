import os
import numpy as np
import cv2
import pytest

# Import functions from your module.
# sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from paz.backend.from_yolo_dataloader import (
    validate_directories,
    get_image_files,
    get_label_files,
    get_image_size,
    load_label,
    process_labels,
    get_data_PAZ_formate,
    validate_file_correspondence,
)


@pytest.fixture
def data_loader_dirs(tmp_path):
    """
    Fixture to set up temporary directories with dummy image and label files.
    """
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image_size = (416, 416)  # (width, height)
    dummy_image = np.full(
        (image_size[1], image_size[0], 3), 255, dtype=np.uint8
    )

    # Create dummy image files.
    cv2.imwrite(str(images_dir / "test1.jpg"), dummy_image)
    cv2.imwrite(str(images_dir / "test2.png"), dummy_image)

    # Create corresponding label files.
    (labels_dir / "test1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (labels_dir / "test2.txt").write_text("1 0.3 0.3 0.1 0.1\n")

    return {
        "test_dir": str(test_dir),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "dummy_image": dummy_image,
        "image_size": image_size,
    }


def test_validate_directories_success(data_loader_dirs):
    images_dir = data_loader_dirs["images_dir"]
    labels_dir = data_loader_dirs["labels_dir"]
    # Should not raise an exception.
    validate_directories(images_dir, labels_dir)


def test_validate_directories_failure(data_loader_dirs):
    labels_dir = data_loader_dirs["labels_dir"]
    with pytest.raises(FileNotFoundError):
        validate_directories("nonexistent/images", labels_dir)
    images_dir = data_loader_dirs["images_dir"]
    with pytest.raises(FileNotFoundError):
        validate_directories(images_dir, "nonexistent/labels")


def test_get_image_files(data_loader_dirs):
    images_dir = data_loader_dirs["images_dir"]
    image_files = get_image_files(images_dir)
    assert "test1.jpg" in image_files
    assert "test2.png" in image_files


def test_get_label_files(data_loader_dirs):
    labels_dir = data_loader_dirs["labels_dir"]
    label_files = get_label_files(labels_dir)
    assert "test1.txt" in label_files
    assert "test2.txt" in label_files


def test_get_image_size(data_loader_dirs):
    images_dir = data_loader_dirs["images_dir"]
    image_path = os.path.join(images_dir, "test1.jpg")
    size = get_image_size(image_path)
    assert size == data_loader_dirs["image_size"]


def test_load_label(data_loader_dirs):
    labels_dir = data_loader_dirs["labels_dir"]
    label_path = os.path.join(labels_dir, "test1.txt")
    labels = load_label(label_path)
    expected = np.array([[0.0, 0.5, 0.5, 0.2, 0.2]])
    np.testing.assert_array_almost_equal(labels, expected)


def test_process_labels(data_loader_dirs):
    image_size = data_loader_dirs["image_size"]
    labels = np.array([[0, 0.5, 0.5, 0.2, 0.2]])
    # Expected absolute coordinates (as per your processing logic).
    expected = np.array([[166, 166, 249, 249, 0]])
    processed = process_labels(labels, image_size, normalize=False)
    np.testing.assert_array_equal(processed, expected)


def test_get_data_PAZ_formate(data_loader_dirs):
    images_dir = data_loader_dirs["images_dir"]
    labels_dir = data_loader_dirs["labels_dir"]
    data = get_data_PAZ_formate(images_dir, labels_dir, normalize=False)
    # Expect 2 data entries (one per image file)
    assert len(data) == 2
    for entry in data:
        assert "image" in entry
        assert "boxes" in entry

    for entry in data:
        base = os.path.splitext(os.path.basename(entry["image"]))[0]
        if base == "test1":
            expected = np.array([[166, 166, 249, 249, 0]])
            np.testing.assert_array_equal(entry["boxes"], expected)
        elif base == "test2":
            expected = np.array([[104, 104, 145, 145, 1]])
            np.testing.assert_array_equal(entry["boxes"], expected)


def test_validate_file_correspondence_failure(data_loader_dirs):
    images_dir = data_loader_dirs["images_dir"]
    dummy_image = data_loader_dirs["dummy_image"]
    # Add an extra image file without a corresponding label file.
    extra_image_path = os.path.join(images_dir, "extra_image.jpg")
    cv2.imwrite(extra_image_path, dummy_image)
    image_files = get_image_files(images_dir)
    labels_dir = data_loader_dirs["labels_dir"]
    label_files = get_label_files(labels_dir)
    with pytest.raises(ValueError) as excinfo:
        validate_file_correspondence(image_files, label_files)
    assert "have no corresponding label files" in str(excinfo.value)


def test_empty_directories(tmp_path):
    empty_images_dir = tmp_path / "empty_images"
    empty_labels_dir = tmp_path / "empty_labels"
    empty_images_dir.mkdir()
    empty_labels_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        get_image_files(str(empty_images_dir))
    with pytest.raises(FileNotFoundError):
        get_label_files(str(empty_labels_dir))


if __name__ == "__main__":
    pytest.main([__file__])
