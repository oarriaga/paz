import jax
import jax.numpy as jp

from paz.inference.metropolis_hastings import (
    Samples,
    apply_gaussian_noise,
    build_new_proposal,
    build_now_proposal,
    choose_proposal,
    propose_additively,
    sample,
)


def test_apply_gaussian_noise_sigma_zero():
    key = jax.random.PRNGKey(0)
    position = jp.array(0.0)
    output = apply_gaussian_noise(key, position, mu=0.0, sigma=0.0)
    assert output == 0.0


def test_build_now_proposal_weight_zero():
    state = Samples(jp.array(0.0), jp.array(0.0))
    proposal = build_now_proposal(state)
    assert proposal.weight == 0.0


def test_build_new_proposal_sum_log_p_accept():
    old_state = Samples(jp.array(0.0), jp.array(0.0))
    new_state = Samples(jp.array(0.0), jp.array(1.0))
    proposal = build_new_proposal(old_state, new_state)
    assert proposal.sum_log_p_accept == 0.0


def test_choose_proposal_rejects_low_weight():
    key = jax.random.PRNGKey(1)
    now_state = Samples(jp.array(0.0), jp.array(0.0))
    new_state = Samples(jp.array(0.0), jp.array(-1e9))
    now = build_now_proposal(now_state)
    new = build_new_proposal(now_state, new_state)
    result = choose_proposal(key, now, new)
    assert result.is_accepted == False


def test_propose_additively_sigma_zero_no_change():
    key = jax.random.PRNGKey(2)
    position = jp.array(1.0)
    proposed = propose_additively(key, position, sigma=0.0)
    assert proposed == 1.0


def test_sample_shapes():
    key = jax.random.PRNGKey(3)
    num_samples = 3
    num_chains = 2
    positions = jp.zeros((num_chains,))

    def log_density_fn(position):
        return -jp.sum(position ** 2)

    states, _ = sample(
        key,
        log_density_fn,
        positions,
        sigma=0.1,
        num_samples=num_samples,
        num_chains=num_chains,
        progress=False,
    )
    assert states.position.shape == (num_samples, num_chains)
