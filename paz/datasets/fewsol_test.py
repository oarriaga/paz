import jax.numpy as jp
import paz
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


def test_masks_to_boxes_matches_non_vectorized():
    masks = jp.array(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    expected = jp.array([paz.mask.to_box(mask, 1.0) for mask in masks])
    expected = paz.boxes.xyxy_to_xywh(expected)
    expected = jp.expand_dims(jp.array(expected), axis=1)
    result = fewsol.masks_to_boxes(masks)
    assert jp.allclose(result, expected)
