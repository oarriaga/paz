import numpy as np
from typing import Literal


def drop_scheduler(
    drop_rate: float,
    epochs: int,
    niter_per_ep: int,
    cutoff_epoch: int = 0,
    mode: Literal['standard', 'early', 'late'] = 'standard',
    schedule: Literal['constant', 'linear'] = 'constant',
) -> np.ndarray:
    """Build a per-iteration drop-path rate schedule.

    Args:
        drop_rate (float): Maximum drop-path rate.
        epochs (int): Total number of training epochs.
        niter_per_ep (int): Iterations per epoch.
        cutoff_epoch (int): Epoch boundary between the *early* and *late*
            phases. Only used when *mode* is ``'early'`` or ``'late'``.
        mode: ``'standard'`` applies *drop_rate* for the entire training.
            ``'early'`` applies it only before *cutoff_epoch* (zero after).
            ``'late'`` applies it only after *cutoff_epoch* (zero before).
        schedule: ``'constant'`` keeps *drop_rate* fixed;
            ``'linear'`` linearly ramps it to zero (only for ``'early'``
            mode).

    Returns:
        np.ndarray: 1-D array of length ``epochs * niter_per_ep`` with the
            drop-path rate for each training iteration.
    """
    assert mode in ['standard', 'early', 'late']
    if mode == 'standard':
        return np.full(epochs * niter_per_ep, drop_rate)

    early_iters = cutoff_epoch * niter_per_ep
    late_iters = (epochs - cutoff_epoch) * niter_per_ep

    if mode == 'early':
        assert schedule in ['constant', 'linear']
        if schedule == 'constant':
            early_schedule = np.full(early_iters, drop_rate)
        elif schedule == 'linear':
            early_schedule = np.linspace(drop_rate, 0, early_iters)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, 0)))
    elif mode == 'late':
        assert schedule in ['constant']
        early_schedule = np.full(early_iters, 0)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, drop_rate)))

    assert len(final_schedule) == epochs * niter_per_ep
    return final_schedule
