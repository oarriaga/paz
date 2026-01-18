import pytest

from paz.inference import metadata
from paz.inference.types import NodeMetadata, PGMMetadata


def test_get_pgm_metadata_missing_raises():
    class Dummy:
        pass

    with pytest.raises(ValueError):
        metadata.get_pgm_metadata(Dummy())


def test_get_node_metadata_missing_raises():
    class Dummy:
        pass

    with pytest.raises(ValueError):
        metadata.get_node_metadata(Dummy())


def _build_pgm_with_metadata():
    meta = PGMMetadata(
        nodes=["n"],
        inputs=["i"],
        non_priors=["np"],
        latent_nodes=["l"],
        output_nodes=["o"],
        observable_nodes=["obs"],
    )

    class Dummy:
        metadata = meta

    return Dummy()


def test_get_inputs_returns_inputs():
    pgm = _build_pgm_with_metadata()
    assert metadata.get_inputs(pgm) == ["i"]


def test_get_non_priors_returns_non_priors():
    pgm = _build_pgm_with_metadata()
    assert metadata.get_non_priors(pgm) == ["np"]


def test_get_latent_nodes_returns_latents():
    pgm = _build_pgm_with_metadata()
    assert metadata.get_latent_nodes(pgm) == ["l"]


def test_get_output_nodes_returns_outputs():
    pgm = _build_pgm_with_metadata()
    assert metadata.get_output_nodes(pgm) == ["o"]


def test_get_observable_nodes_returns_observables():
    pgm = _build_pgm_with_metadata()
    assert metadata.get_observable_nodes(pgm) == ["obs"]


def test_get_nodes_returns_nodes():
    pgm = _build_pgm_with_metadata()
    assert metadata.get_nodes(pgm) == ["n"]


def test_get_distribution_fn_returns_value():
    class Dummy:
        metadata = NodeMetadata(lambda: None, None)

    assert metadata.get_distribution_fn(Dummy()) is not None


def test_get_bijector_returns_value():
    class Dummy:
        metadata = NodeMetadata(None, "bijector")

    assert metadata.get_bijector(Dummy()) == "bijector"
