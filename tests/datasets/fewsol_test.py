import jax.numpy as jp
import paz.datasets.fewsol as fewsol


def test_ensure_dataset_returns_existing_root(tmp_path, monkeypatch):
    dataset_root = tmp_path / "real_objects"
    dataset_root.mkdir()
    called = {"value": False}

    def fake_get_file(*args, **kwargs):
        called["value"] = True
        return str(tmp_path)

    monkeypatch.setattr(fewsol, "get_file", fake_get_file)
    result = fewsol.ensure_dataset(dataset_root)
    assert result == dataset_root
    assert called["value"] is False


def test_ensure_dataset_downloads_when_missing(tmp_path, monkeypatch):
    dataset_root = tmp_path / "real_objects"
    download_root = tmp_path / "FEWSOL"
    expected = download_root / "real_objects"
    expected.mkdir(parents=True)

    def fake_get_file(*args, **kwargs):
        return str(download_root)

    monkeypatch.setattr(fewsol, "get_file", fake_get_file)
    result = fewsol.ensure_dataset(dataset_root)
    assert result == expected


def test_masks_to_boxes_returns_one_xywh_box_per_mask():
    masks = jp.zeros((2, 6, 8)).at[0, 1:3, 2:6].set(1.0).at[1, 4, 7].set(1.0)
    expected = jp.array([[[2, 1, 3, 1]], [[7, 4, 0, 0]]])
    assert jp.array_equal(fewsol.masks_to_boxes(masks), expected)


def test_masks_to_boxes_accepts_a_trailing_channel():
    masks = jp.zeros((1, 6, 8, 1)).at[0, 1:3, 2:6].set(1.0)
    expected = jp.array([[[2, 1, 3, 1]]])
    assert jp.array_equal(fewsol.masks_to_boxes(masks), expected)
