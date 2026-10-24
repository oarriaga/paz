import numpy as np
import pytest

from paz.models.foundation.kimodo import features, pretrained


def write_assets(directory):
    dims = {"global_root": features.GLOBAL_ROOT_DIM,
            "local_root": features.LOCAL_ROOT_DIM, "body": features.BODY_DIM}
    arrays = {}
    for block, dim in dims.items():
        arrays[f"{block}_mean"] = np.zeros(dim, "float32")
        arrays[f"{block}_std"] = np.ones(dim, "float32")
    np.savez(directory / pretrained.ASSETS_NAME, **arrays)
    np.savez(directory / pretrained.PROMPTS_NAME,
             **{"A person walks forward.": np.ones((1, 4096), "float32")})


def test_builds_released_architecture_from_local_assets(tmp_path):
    write_assets(tmp_path)
    kimodo = pretrained.build_from_assets(tmp_path)
    # Parameter count of the released safetensors checkpoint.
    assert kimodo.denoiser.count_params() == 283_527_585
    assert kimodo.stats.body.mean.shape == (features.BODY_DIM,)


def test_download_fetches_weights_and_assets(tmp_path, monkeypatch):
    write_assets(tmp_path)
    requested = []

    def fake_get_file(name, url, cache_subdir):
        requested.append((name, url))
        return str(tmp_path / name)

    monkeypatch.setattr(pretrained, "get_file", fake_get_file)
    assert pretrained.resolve_dir(None) == tmp_path
    names = [name for name, _ in requested]
    assert names == [pretrained.WEIGHTS_NAME, pretrained.ASSETS_NAME]
    assert all(url.startswith(pretrained.WEIGHTS_URL)
               for _, url in requested)
    prompts = pretrained.fetch_prompt_embeddings()
    assert set(prompts) == {"A person walks forward."}


def test_missing_prompt_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pretrained.fetch_prompt_embeddings(tmp_path)
