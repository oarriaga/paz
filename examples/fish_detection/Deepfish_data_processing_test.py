import os
import zipfile
import pytest

# Import functions from your module.
from examples.fish_detection.Deepfish_data_processing import (
    get_file,
    get_gdrive_file_id,
    extract_compresed_file,
    process_folder,
    create_output_directories,
    write_classes_file,
    process_class_folders,
    process_negative_samples,
    process_class_data,
)


def test_process_folder(tmp_path, capsys):
    """
    Test that process_folder copies images and corresponding label files.
    It should also warn if a label file is missing.
    """
    # Setup source and destination directories.
    src = tmp_path / "src"
    dst_img = tmp_path / "dst_img"
    dst_label = tmp_path / "dst_label"
    src.mkdir(), dst_img.mkdir(), dst_label.mkdir()

    # Create two image files.
    (src / "image1.jpg").write_bytes(b"dummy image content")
    (src / "image2.png").write_bytes(b"dummy image content")
    # Create label for image1 only.
    (src / "image1.txt").write_text("label content")

    # Process the folder.
    process_folder(str(src), str(dst_img), str(dst_label))

    # Check that both images were copied.
    assert (dst_img / "image1.jpg").exists()
    assert (dst_img / "image2.png").exists()

    # Check that the label for image1 was copied, but not for image2.
    assert (dst_label / "image1.txt").exists()
    assert not (dst_label / "image2.txt").exists()

    # Verify that a warning was printed for the missing label.
    captured = capsys.readouterr().out
    assert "Warning: No label found for image image2.png" in captured


def test_create_output_directories(tmp_path):
    """
    Test that create_output_directories creates the required subdirectories.
    """
    output_path = tmp_path / "output"
    output_path.mkdir()

    create_output_directories(str(output_path))

    for split in ["train", "valid"]:
        assert (output_path / "images" / split).is_dir()
        assert (output_path / "labels" / split).is_dir()


def test_write_classes_file(tmp_path):
    """
    Test that write_classes_file creates a classes.txt file with the correct content.
    """
    output_path = tmp_path / "output"
    output_path.mkdir()

    write_classes_file(str(output_path))

    classes_file = output_path / "classes.txt"
    assert classes_file.exists()
    assert classes_file.read_text() == "Fish\n"


def test_process_class_folders(tmp_path, capsys):
    """
    Test process_class_folders by creating a fake raw data directory with a digit-named
    class folder containing train and valid splits.
    """
    # Create raw_data directory and a class folder "0".
    raw_data = tmp_path / "raw_data"
    raw_data.mkdir()
    class0 = raw_data / "0"
    class0.mkdir()

    # Create "train" folder with one image and its label.
    train_folder = class0 / "train"
    train_folder.mkdir()
    (train_folder / "train_img.jpg").write_bytes(b"image content")
    (train_folder / "train_img.txt").write_text("train label")

    # Create "valid" folder with one image but no label (to trigger warning).
    valid_folder = class0 / "valid"
    valid_folder.mkdir()
    (valid_folder / "valid_img.png").write_bytes(b"image content")

    # Also create a non-digit folder that should be ignored.
    non_digit = raw_data / "not_a_digit"
    non_digit.mkdir()
    (non_digit / "dummy.jpg").write_bytes(b"dummy")

    # Setup destination output structure.
    output_path = tmp_path / "output"
    output_path.mkdir()
    create_output_directories(str(output_path))

    process_class_folders(str(raw_data), str(output_path))

    # Verify that the train image and label were copied.
    assert (output_path / "images" / "train" / "train_img.jpg").exists()
    assert (output_path / "labels" / "train" / "train_img.txt").exists()

    # Verify that the valid image was copied and label was not.
    assert (output_path / "images" / "valid" / "valid_img.png").exists()
    assert not (output_path / "labels" / "valid" / "valid_img.txt").exists()

    captured = capsys.readouterr().out
    assert "Warning: No label found for image valid_img.png" in captured


def test_process_negative_samples(tmp_path, capsys):
    """
    Test process_negative_samples by creating a Negative_samples folder with train and valid splits.
    """
    raw_data = tmp_path / "raw_data"
    raw_data.mkdir()
    neg_samples = raw_data / "Negative_samples"
    neg_samples.mkdir()

    # Create train folder with an image and its label.
    train_folder = neg_samples / "train"
    train_folder.mkdir()
    (train_folder / "neg_train.jpg").write_bytes(b"image content")
    (train_folder / "neg_train.txt").write_text("neg label")

    # Create valid folder with an image missing its label.
    valid_folder = neg_samples / "valid"
    valid_folder.mkdir()
    (valid_folder / "neg_valid.jpg").write_bytes(b"image content")

    # Setup output structure.
    output_path = tmp_path / "output"
    output_path.mkdir()
    create_output_directories(str(output_path))

    process_negative_samples(str(raw_data), str(output_path))

    # Check train split: both image and label should be copied.
    assert (output_path / "images" / "train" / "neg_train.jpg").exists()
    assert (output_path / "labels" / "train" / "neg_train.txt").exists()

    # Check valid split: image should be copied, label should be missing.
    assert (output_path / "images" / "valid" / "neg_valid.jpg").exists()
    assert not (output_path / "labels" / "valid" / "neg_valid.txt").exists()

    captured = capsys.readouterr().out
    assert "Warning: No label found for image neg_valid.jpg" in captured


def test_process_class_data(tmp_path, capsys):
    """
    Test process_class_data which orchestrates the entire workflow.
    It should create output directories, write classes.txt, process class folders,
    and process negative samples.
    """
    # Create a fake raw_data structure.
    raw_data = tmp_path / "raw_data"
    raw_data.mkdir()

    # Create a class folder "0" with train and valid splits.
    class0 = raw_data / "0"
    class0.mkdir()
    train_folder = class0 / "train"
    train_folder.mkdir()
    (train_folder / "class_train.jpg").write_bytes(b"image content")
    (train_folder / "class_train.txt").write_text("class train label")
    valid_folder = class0 / "valid"
    valid_folder.mkdir()
    (valid_folder / "class_valid.png").write_bytes(b"image content")
    (valid_folder / "class_valid.txt").write_text("class valid label")

    # Create Negative_samples with train and valid splits.
    neg_samples = raw_data / "Negative_samples"
    neg_samples.mkdir()
    neg_train = neg_samples / "train"
    neg_train.mkdir()
    (neg_train / "neg_train.jpg").write_bytes(b"image content")
    (neg_train / "neg_train.txt").write_text("neg train label")
    neg_valid = neg_samples / "valid"
    neg_valid.mkdir()
    (neg_valid / "neg_valid.jpg").write_bytes(b"image content")
    # Note: no label for neg_valid.jpg to trigger a warning.

    # Setup output directory.
    output_path = tmp_path / "output"
    output_path.mkdir()

    # Run the entire processing workflow.
    process_class_data(str(raw_data), str(output_path))

    # Check that the necessary directories were created.
    for split in ["train", "valid"]:
        assert (output_path / "images" / split).is_dir()
        assert (output_path / "labels" / split).is_dir()

    # Check that classes.txt was created correctly.
    classes_file = output_path / "classes.txt"
    assert classes_file.exists()
    assert classes_file.read_text() == "Fish\n"

    # Verify that class folder files are copied.
    assert (output_path / "images" / "train" / "class_train.jpg").exists()
    assert (output_path / "labels" / "train" / "class_train.txt").exists()
    assert (output_path / "images" / "valid" / "class_valid.png").exists()
    assert (output_path / "labels" / "valid" / "class_valid.txt").exists()

    # Verify that negative sample files are copied.
    assert (output_path / "images" / "train" / "neg_train.jpg").exists()
    assert (output_path / "labels" / "train" / "neg_train.txt").exists()
    assert (output_path / "images" / "valid" / "neg_valid.jpg").exists()
    assert not (output_path / "labels" / "valid" / "neg_valid.txt").exists()

    captured = capsys.readouterr().out
    assert "Warning: No label found for image neg_valid.jpg" in captured


def test_get_gdrive_file_id_valid_d_link():
    link = "https://drive.google.com/file/d/FILEID12345/view?usp=sharing"
    expected = "FILEID12345"
    result = get_gdrive_file_id(link)
    assert result == expected


def test_get_gdrive_file_id_valid_id_param():
    link = "https://drive.google.com/open?id=FILEID67890"
    expected = "FILEID67890"
    result = get_gdrive_file_id(link)
    assert result == expected


def test_get_gdrive_file_id_invalid_link():
    link = "https://example.com/file/abc"
    with pytest.raises(ValueError):
        get_gdrive_file_id(link)


def test_get_file_gdrive(monkeypatch, tmp_path):
    fake_file_url = "https://drive.google.com/file/d/FAKEID/view?usp=sharing"
    output_filename = "fake_file.txt"
    output_dir = tmp_path / "download"
    output_dir.mkdir()

    # Patch get_gdrive_file_id to return "FAKEID"
    monkeypatch.setattr(
        "examples.fish_detection.Deepfish_data_processing.get_gdrive_file_id",
        lambda url: "FAKEID",
    )

    # Patch gdown.download with a fake function that writes content.
    def fake_download(url, out_path, quiet):
        with open(out_path, "w") as f:
            f.write("fake content")

    monkeypatch.setattr("gdown.download", fake_download)

    result = get_file(
        fake_file_url, output_filename, output_dir=str(output_dir)
    )
    expected_path = os.path.join(str(output_dir), output_filename)
    assert result == expected_path
    with open(result, "r") as f:
        content = f.read()
    assert content == "fake content"


def test_get_file_regular_url(monkeypatch):
    fake_file_url = "https://example.com/fake_file.txt"
    output_filename = "fake_file.txt"
    fake_local_path = os.path.join("/fake", "path", output_filename)

    def fake_keras_get_file(fname, origin, cache_dir):
        return fake_local_path

    monkeypatch.setattr("tensorflow.keras.utils.get_file", fake_keras_get_file)

    result = get_file(fake_file_url, output_filename, output_dir="/fake/path")
    assert result == fake_local_path


def test_extract_compressed_file(tmp_path):
    # Create a temporary ZIP file with one text file inside.
    zip_file_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    inner_filename = "test.txt"
    inner_file_content = "Hello, world!"
    temp_file_path = tmp_path / inner_filename
    temp_file_path.write_text(inner_file_content)

    with zipfile.ZipFile(str(zip_file_path), "w") as zipf:
        zipf.write(str(temp_file_path), arcname=inner_filename)

    result_dir = extract_compresed_file(
        str(zip_file_path), output_dir=str(extract_dir)
    )
    assert result_dir == str(extract_dir)

    extracted_file_path = extract_dir / inner_filename
    assert extracted_file_path.exists()
    assert extracted_file_path.read_text() == inner_file_content


if __name__ == "__main__":
    pytest.main([__file__])
