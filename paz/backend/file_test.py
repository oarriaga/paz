import os
import json
import tempfile

import pytest

from paz.backend import file


def test_write_json_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")
        test_dict = {"key": "value", "number": 42}

        file.write_json(test_dict, filepath)

        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded == test_dict


def test_write_json_uses_custom_indent():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")
        test_dict = {"key": "value"}

        file.write_json(test_dict, filepath, indent=2)

        with open(filepath, "r") as f:
            content = f.read()
        assert "  " in content


def test_write_json_overwrites_existing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")

        file.write_json({"old": "data"}, filepath)
        file.write_json({"new": "data"}, filepath)

        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded == {"new": "data"}


def test_load_csv_parses_simple_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.csv")
        csv_content = "epoch,loss,accuracy\n1,0.5,0.8\n2,0.3,0.9\n"

        with open(filepath, "w") as f:
            f.write(csv_content)

        result = file.load_csv(filepath)

        assert result["epoch"] == [1, 2]
        assert result["loss"] == [0.5, 0.3]
        assert result["accuracy"] == [0.8, 0.9]


def test_load_csv_handles_whitespace_in_headers():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.csv")
        csv_content = " epoch , loss , accuracy \n1,0.5,0.8\n"

        with open(filepath, "w") as f:
            f.write(csv_content)

        result = file.load_csv(filepath)

        assert "epoch" in result
        assert "loss" in result
        assert " epoch " not in result


def test_load_csv_raises_on_missing_file():
    with pytest.raises(FileNotFoundError, match="was not found"):
        file.load_csv("/nonexistent/path.csv")


def test_load_csv_raises_on_column_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.csv")
        csv_content = "epoch,loss\n1,0.5\n2,0.3,0.9\n"

        with open(filepath, "w") as f:
            f.write(csv_content)

        with pytest.raises(ValueError, match="Invalid column size"):
            file.load_csv(filepath)
