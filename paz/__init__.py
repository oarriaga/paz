from paz import datasets
from paz.backend import draw
from paz.backend import boxes
from paz.backend import image
from paz.backend import classes
from paz.backend import pinhole
from paz.backend import mask
from paz.backend import pointcloud
from paz.backend import detection
from paz.backend import standard
from paz.backend import depth
from paz.backend import logger  # DEPRECATED - use directory, file, message
from paz.backend import directory
from paz.backend import file
from paz.backend import message
from paz.backend import log
from paz.backend.camera import Camera, VideoPlayer
from paz.backend.lie import SE3
from paz.backend.lie import SO3
from paz.backend.lie import SE2
from paz.backend.lie import quaternion

from paz import graphics
from paz.backend import points2D
from paz.backend import algebra
from paz.backend import scene
from paz.backend import plane
from paz.backend.standard import (
    lock,
    partial,
    merge_dicts,
    cast,
    to_jax,
    to_numpy,
    NamedTuple,
)
from paz import losses
from paz.abstract import Model, Node, Input, Sequential, Tree
from paz.models.decomposition import pca as PCA
from paz import applications
from paz import utils
from paz.utils import pytree
from paz.utils import assert_snapshot, cache, jit_and_cache, clear_cache
from paz.utils import CACHE_PATH
from paz.abstract import tree
from paz import callbacks
from paz import layers
from paz import minimization
from paz import optimization
from paz import optimizers
from paz.minimization import minimize

from paz import inference
from paz.inference import metropolis_hastings
from paz.inference.prior import Prior
from paz.inference.observable import Observable
from paz.inference.latent import Latent
from paz.inference.pgm import PGM
from paz.inference.tuner import AdaptiveStepTuner, Tuner
from paz.inference.pgm import marginalize, recover_discrete_posterior
from paz.inference.infer import infer
from paz.inference.posterior import MCMCPosterior
from paz.inference.metadata import (
    get_bijector,
    get_distribution_fn,
    get_inputs,
    get_latent_nodes,
    get_nodes,
    get_non_priors,
)


def __getattr__(name):
    if name == "time":
        return utils.time
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
