import jax
import jax.numpy as jp
import paz


def test_STN_builds_and_runs_with_defaults():
    model = paz.models.STN()
    images = jax.random.uniform(jax.random.PRNGKey(0), (2, 40, 40, 1))
    scores = jax.jit(model)(images)
    assert scores.shape == (2, 10)
    assert jp.all(jp.isfinite(scores))
    assert jp.allclose(jp.sum(scores, axis=1), 1.0, atol=1e-5)
