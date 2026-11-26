import json
import paz.datasets.fsclvr as fsclvr


def test_parse_metadata_defaults_to_plain_scene(tmp_path, monkeypatch):
    scene_path = tmp_path / "datasets" / "PRIMITIVES"
    scene_path.mkdir(parents=True)
    metadata = {"meshes": ["cube", "sphere"]}
    (scene_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(fsclvr, "get_file", lambda *_, **__: str(tmp_path))
    assert fsclvr.parse_metadata() == metadata
