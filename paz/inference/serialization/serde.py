import jax
from jax import flatten_util

from paz.inference.density.core import _build_distribution_density
from paz.inference.metropolis_hastings import Samples, State
from paz.inference.posterior import MCMCPosterior, MCMCPosteriorType
from paz.inference.pgm import PGM
from paz.inference.prior import Prior
from paz.inference.types import Density, PGMMetadata

from .graph import _build_graph, _serialize_graph
from .io import _build_manifest
from .spec import (
    _build_latent_space,
    _clean_config,
    _deserialize_bijector,
    _deserialize_distribution,
    _deserialize_distribution_obj,
    _dict_to_samples,
    _dummy_sample,
    _latent_space_spec,
    _ref,
    _resolve_ref,
    _sample_spec,
    _samples_to_dict,
    _serialize_bijector,
    _serialize_distribution,
    _serialize_distribution_obj,
)


class PosteriorSerde:
    type_id = "Posterior"

    def can_handle(self, obj):
        return isinstance(obj, MCMCPosteriorType)

    def to_spec(self, obj):
        arrays = {}
        position, sample_kind = _samples_to_dict(obj.samples.position)
        position_refs = {}
        for name, value in position.items():
            array_name = f"position_{name}"
            arrays[array_name] = value
            position_refs[name] = _ref(array_name)
        arrays["log_density"] = obj.samples.log_density
        infos = obj.infos
        arrays["is_accepted"] = infos.is_accepted
        arrays["acceptance_rate"] = infos.acceptance_rate
        latent_spec = _latent_space_spec(obj.latent_space, arrays, "latent")
        payload = {
            "posterior_kind": "mcmc",
            "method": obj.config.get("method", "mh"),
            "config": _clean_config(obj.config),
            "latent_space": latent_spec,
            "sample_kind": sample_kind,
            "samples": {
                "position": position_refs,
                "log_density": _ref("log_density"),
            },
            "infos": {
                "is_accepted": _ref("is_accepted"),
                "acceptance_rate": _ref("acceptance_rate"),
            },
        }
        manifest = _build_manifest(self.type_id, 1)
        return manifest, payload, arrays

    def from_spec(self, manifest, payload, arrays):
        latent_space = _build_latent_space(payload["latent_space"], arrays)
        position = {
            name: _resolve_ref(ref, arrays)
            for name, ref in payload["samples"]["position"].items()
        }
        position = _dict_to_samples(
            position, payload["sample_kind"], latent_space
        )
        log_density = _resolve_ref(payload["samples"]["log_density"], arrays)
        samples = Samples(position, log_density)
        is_accepted = _resolve_ref(payload["infos"]["is_accepted"], arrays)
        acceptance_rate = _resolve_ref(
            payload["infos"]["acceptance_rate"], arrays
        )
        infos = State(None, is_accepted, acceptance_rate)
        config = payload.get("config", {})
        space = config.get("space", "inv")
        return MCMCPosterior(samples, infos, config, latent_space, space)


class DensitySerde:
    type_id = "Density"

    def can_handle(self, obj):
        return isinstance(obj, Density)

    def to_spec(self, obj):
        arrays = {}
        spec = _serialize_distribution(obj, arrays)
        sample = obj.sample(jax.random.PRNGKey(0), num_samples=1, space="inv")
        sample_spec = _sample_spec(sample)
        latent_spec = _latent_space_spec(obj.latent_space, arrays, "latent")
        payload = {
            "density": spec,
            "sample_spec": sample_spec,
            "latent_space": latent_spec,
        }
        manifest = _build_manifest(self.type_id, 1)
        return manifest, payload, arrays

    def from_spec(self, manifest, payload, arrays):
        latent_space = _build_latent_space(payload["latent_space"], arrays)
        dummy = _dummy_sample(payload["sample_spec"], latent_space)
        _, unravel = flatten_util.ravel_pytree(dummy)
        distribution = _deserialize_distribution(payload["density"], arrays)
        return _build_distribution_density(distribution, latent_space, unravel)


class PriorSerde:
    type_id = "Prior"

    def can_handle(self, obj):
        return getattr(obj, "distribution", None) is not None

    def to_spec(self, obj):
        arrays = {}
        spec = _serialize_distribution_obj(obj.distribution, arrays, obj.name)
        bijector = _serialize_bijector(
            obj.metadata.bijector, arrays, f"{obj.name}_bijector"
        )
        payload = {
            "name": obj.name,
            "distribution": spec,
            "bijector": bijector,
        }
        manifest = _build_manifest(self.type_id, 1)
        return manifest, payload, arrays

    def from_spec(self, manifest, payload, arrays):
        distribution = _deserialize_distribution_obj(
            payload["distribution"], arrays
        )
        bijector = _deserialize_bijector(payload["bijector"], arrays)
        return Prior(distribution, bijector=bijector, name=payload["name"])


class LatentSerde:
    type_id = "Latent"

    def can_handle(self, obj):
        if isinstance(getattr(obj, "metadata", None), PGMMetadata):
            return False
        return obj.sample_inverse is not None and obj.distribution is None

    def to_spec(self, obj):
        arrays = {}
        graph = _serialize_graph([obj], arrays)
        payload = {"root": obj.name, "graph": graph}
        manifest = _build_manifest(self.type_id, 1)
        return manifest, payload, arrays

    def from_spec(self, manifest, payload, arrays):
        nodes = _build_graph(payload["graph"], arrays)
        return nodes[payload["root"]]


class ObservableSerde:
    type_id = "Observable"

    def can_handle(self, obj):
        if isinstance(getattr(obj, "metadata", None), PGMMetadata):
            return False
        return obj.sample_inverse is None and obj.distribution is None

    def to_spec(self, obj):
        arrays = {}
        graph = _serialize_graph([obj], arrays)
        payload = {"root": obj.name, "graph": graph}
        manifest = _build_manifest(self.type_id, 1)
        return manifest, payload, arrays

    def from_spec(self, manifest, payload, arrays):
        nodes = _build_graph(payload["graph"], arrays)
        return nodes[payload["root"]]


class PGMSerde:
    type_id = "PGM"

    def can_handle(self, obj):
        return isinstance(getattr(obj, "metadata", None), PGMMetadata)

    def to_spec(self, obj):
        arrays = {}
        metadata = obj.metadata
        graph = _serialize_graph(metadata.output_nodes, arrays)
        payload = {
            "name": obj.name,
            "graph": graph,
            "inputs": [node.name for node in metadata.inputs],
            "outputs": [node.name for node in metadata.output_nodes],
        }
        manifest = _build_manifest(self.type_id, 1)
        return manifest, payload, arrays

    def from_spec(self, manifest, payload, arrays):
        nodes = _build_graph(payload["graph"], arrays)
        inputs = [nodes[name] for name in payload["inputs"]]
        outputs = [nodes[name] for name in payload["outputs"]]
        return PGM(inputs, outputs, payload["name"])


SERDE_REGISTRY = [
    PosteriorSerde(),
    DensitySerde(),
    PriorSerde(),
    LatentSerde(),
    ObservableSerde(),
    PGMSerde(),
]


def _find_serde(obj):
    for serde in SERDE_REGISTRY:
        if serde.can_handle(obj):
            return serde
    if hasattr(obj, "_fields"):
        raise ValueError(
            "This object contains unserializable callables. Use "
            "a serializable spec."
        )
    raise ValueError("No serializer available for this object.")


def _serde_for_type(type_id):
    for serde in SERDE_REGISTRY:
        if serde.type_id == type_id:
            return serde
    raise ValueError(f"No serializer registered for '{type_id}'.")
