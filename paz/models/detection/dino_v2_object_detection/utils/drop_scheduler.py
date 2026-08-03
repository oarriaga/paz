import numpy as np

DROP_MODES = ("standard", "early", "late")


def build_constant_schedule(drop_rate, early_steps, late_steps, mode):
    if mode == "early":
        head, tail = np.full(early_steps, drop_rate), np.full(late_steps, 0)
    else:
        head, tail = np.full(early_steps, 0), np.full(late_steps, drop_rate)
    return np.concatenate((head, tail))


def build_linear_schedule(drop_rate, early_steps, late_steps):
    head = np.linspace(drop_rate, 0, early_steps)
    return np.concatenate((head, np.full(late_steps, 0)))


def drop_scheduler(drop_rate, epochs, num_steps_per_epoch, cutoff_epoch=0, mode='standard', schedule='constant'):  # fmt: skip
    assert mode in DROP_MODES
    total_steps = epochs * num_steps_per_epoch
    early_steps = cutoff_epoch * num_steps_per_epoch
    late_steps = (epochs - cutoff_epoch) * num_steps_per_epoch
    if mode == "standard":
        final_schedule = np.full(total_steps, drop_rate)
    elif schedule == "linear":
        assert mode == "early"
        args = (drop_rate, early_steps, late_steps)
        final_schedule = build_linear_schedule(*args)
        assert len(final_schedule) == total_steps
    else:
        assert schedule == "constant"
        args = (drop_rate, early_steps, late_steps, mode)
        final_schedule = build_constant_schedule(*args)
        assert len(final_schedule) == total_steps
    return final_schedule
