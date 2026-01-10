import os
import tempfile
import time
from pathlib import Path

from paz.backend import directory


def test_make_creates_simple_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_dir")
        result = directory.make(test_path)
        assert os.path.exists(test_path)
        assert os.path.isdir(test_path)
        assert result == test_path


def test_make_creates_nested_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "parent", "child", "grandchild")
        result = directory.make(test_path)
        assert os.path.exists(test_path)
        assert result == test_path


def test_make_is_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_dir")
        directory.make(test_path)
        result = directory.make(test_path)
        assert os.path.exists(test_path)
        assert result == test_path


def test_make_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_dir"
        result = directory.make(test_path)
        assert os.path.exists(test_path)
        assert result == str(test_path)


def test_make_timestamped_creates_directory_with_timestamp():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = directory.make_timestamped(tmpdir, None)
        assert os.path.exists(result)
        assert os.path.isdir(result)
        dirname = os.path.basename(result)
        assert len(dirname) == 19


def test_make_timestamped_includes_label():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = directory.make_timestamped(tmpdir, "experiment")
        assert os.path.exists(result)
        dirname = os.path.basename(result)
        assert "experiment" in dirname


def test_make_timestamped_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = directory.make_timestamped(Path(tmpdir), "experiment")
        assert os.path.exists(result)


def test_find_latest_returns_most_recent_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = os.path.join(tmpdir, "test_01")
        dir2 = os.path.join(tmpdir, "test_02")
        dir3 = os.path.join(tmpdir, "test_03")

        os.makedirs(dir1)
        time.sleep(0.01)
        os.makedirs(dir2)
        time.sleep(0.01)
        os.makedirs(dir3)

        wildcard = os.path.join(tmpdir, "test_*")
        result = directory.find_latest(wildcard)
        assert result == dir3


def test_find_latest_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = Path(tmpdir) / "test_01"
        dir2 = Path(tmpdir) / "test_02"
        dir3 = Path(tmpdir) / "test_03"

        dir1.mkdir()
        time.sleep(0.01)
        dir2.mkdir()
        time.sleep(0.01)
        dir3.mkdir()

        wildcard = Path(tmpdir) / "test_*"
        result = directory.find_latest(wildcard)
        assert result == str(dir3)


def test_find_latest_ignores_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test_dir")
        test_file = os.path.join(tmpdir, "test_file")

        os.makedirs(test_dir)
        Path(test_file).touch()

        wildcard = os.path.join(tmpdir, "test_*")
        result = directory.find_latest(wildcard)
        assert result == test_dir


def test_exists_returns_true_for_existing_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert directory.exists(tmpdir)


def test_exists_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert directory.exists(Path(tmpdir))


def test_exists_returns_false_for_nonexistent_directory():
    assert not directory.exists("/nonexistent/path/to/directory")


def test_exists_returns_false_for_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "file.txt")
        Path(test_file).touch()
        assert not directory.exists(test_file)


def test_is_empty_returns_true_for_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "empty")
        os.makedirs(test_dir)
        assert directory.is_empty(test_dir)


def test_is_empty_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "empty"
        test_dir.mkdir()
        assert directory.is_empty(test_dir)


def test_is_empty_returns_false_for_directory_with_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "file.txt")
        Path(test_file).touch()
        assert not directory.is_empty(tmpdir)


def test_is_empty_returns_false_for_directory_with_subdirectories():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        assert not directory.is_empty(tmpdir)


def test_list_subdirectories_returns_all_subdirectories():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = os.path.join(tmpdir, "dir1")
        dir2 = os.path.join(tmpdir, "dir2")
        dir3 = os.path.join(tmpdir, "dir3")

        os.makedirs(dir1)
        os.makedirs(dir2)
        os.makedirs(dir3)

        result = directory.list_subdirectories(tmpdir)
        assert len(result) == 3
        assert dir1 in result
        assert dir2 in result
        assert dir3 in result


def test_list_subdirectories_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = Path(tmpdir) / "dir1"
        dir2 = Path(tmpdir) / "dir2"

        dir1.mkdir()
        dir2.mkdir()

        result = directory.list_subdirectories(Path(tmpdir))
        assert str(dir1) in result
        assert str(dir2) in result


def test_list_subdirectories_ignores_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        test_file = os.path.join(tmpdir, "file.txt")

        os.makedirs(subdir)
        Path(test_file).touch()

        result = directory.list_subdirectories(tmpdir)
        assert len(result) == 1
        assert subdir in result


def test_list_subdirectories_returns_empty_list_for_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = directory.list_subdirectories(tmpdir)
        assert result == []


def test_list_files_returns_all_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(tmpdir, "file2.txt")
        file3 = os.path.join(tmpdir, "file3.txt")

        Path(file1).touch()
        Path(file2).touch()
        Path(file3).touch()

        result = directory.list_files(tmpdir)
        assert len(result) == 3
        assert file1 in result
        assert file2 in result
        assert file3 in result


def test_list_files_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "file1.txt"
        file2 = Path(tmpdir) / "file2.txt"

        file1.touch()
        file2.touch()

        result = directory.list_files(Path(tmpdir))
        assert str(file1) in result
        assert str(file2) in result


def test_list_files_filters_by_pattern():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(tmpdir, "file2.py")
        file3 = os.path.join(tmpdir, "file3.txt")

        Path(file1).touch()
        Path(file2).touch()
        Path(file3).touch()

        result = directory.list_files(tmpdir, "*.txt")
        assert len(result) == 2
        assert file1 in result
        assert file3 in result
        assert file2 not in result


def test_list_files_ignores_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "file.txt")
        subdir = os.path.join(tmpdir, "subdir")

        Path(test_file).touch()
        os.makedirs(subdir)

        result = directory.list_files(tmpdir)
        assert len(result) == 1
        assert test_file in result


def test_list_files_returns_empty_list_for_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = directory.list_files(tmpdir)
        assert result == []


def test_remove_deletes_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "to_remove")
        os.makedirs(test_dir)

        directory.remove(test_dir)
        assert not os.path.exists(test_dir)


def test_remove_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "to_remove"
        test_dir.mkdir()

        directory.remove(test_dir)
        assert not test_dir.exists()


def test_remove_deletes_directory_with_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "to_remove")
        os.makedirs(test_dir)
        test_file = os.path.join(test_dir, "file.txt")
        Path(test_file).touch()

        directory.remove(test_dir)
        assert not os.path.exists(test_dir)


def test_remove_deletes_nested_directory_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "to_remove")
        nested_dir = os.path.join(test_dir, "nested", "deep")
        os.makedirs(nested_dir)
        test_file = os.path.join(nested_dir, "file.txt")
        Path(test_file).touch()

        directory.remove(test_dir)
        assert not os.path.exists(test_dir)


def test_size_returns_zero_for_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "empty")
        os.makedirs(test_dir)

        result = directory.size(test_dir)
        assert result == 0


def test_size_accepts_path_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "empty"
        test_dir.mkdir()

        result = directory.size(test_dir)
        assert result == 0


def test_size_returns_file_size_for_directory_with_one_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "file.txt")
        content = "Hello, World!"
        with open(test_file, "w") as f:
            f.write(content)

        result = directory.size(tmpdir)
        assert result == len(content)


def test_size_returns_total_size_for_directory_with_multiple_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(tmpdir, "file2.txt")
        content1 = "Hello"
        content2 = "World"

        with open(file1, "w") as f:
            f.write(content1)
        with open(file2, "w") as f:
            f.write(content2)

        result = directory.size(tmpdir)
        assert result == len(content1) + len(content2)


def test_size_includes_nested_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, "nested")
        os.makedirs(nested_dir)
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(nested_dir, "file2.txt")
        content1 = "Hello"
        content2 = "World"

        with open(file1, "w") as f:
            f.write(content1)
        with open(file2, "w") as f:
            f.write(content2)

        result = directory.size(tmpdir)
        assert result == len(content1) + len(content2)
