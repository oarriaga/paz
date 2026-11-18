from paz.inference.pgm.marginalize import marginalize
from paz.inference.pgm.marginalize import recover_discrete_posterior
from paz.inference.pgm.pgm import PGM
from paz.inference.pgm.builder import build_graph

__all__ = ["PGM", "build_graph", "marginalize", "recover_discrete_posterior"]
