import csv
from collections import namedtuple
from pathlib import Path

import jax

FIELDS = "iteration, reward, episode_return, tracking, level, divergences, loss, value_loss, entropy, KL, gradient_norm, learning_rate, max_speed, steps_per_second"  # fmt: skip
Row = namedtuple("Row", FIELDS)


def open_log(directory, name="training.csv"):
    path = Path(directory) / name
    exists = path.exists()
    log_file = path.open("a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=Row._fields)
    if not exists:
        writer.writeheader()
    return log_file, writer


def write_row(writer, log_file, row):
    writer.writerow(row._asdict())
    log_file.flush()


def build_row(iteration, metrics, update_metrics, max_speed, steps, elapsed):
    update = jax.tree.map(float, update_metrics)
    args = iteration, float(metrics.reward), float(metrics.episode_return), float(metrics.terms[0]), float(metrics.level), int(metrics.divergences), update.loss, update.value_loss, update.entropy, update.KL, update.gradient_norm, update.learning_rate, float(max_speed), steps / elapsed  # fmt: skip
    return Row(*args)


def print_row(row):
    message = (f"iteration {row.iteration:5d}"
               f" | reward {row.reward:+.4f}"
               f" | return {row.episode_return:+9.2f}"
               f" | tracking {row.tracking:.3f}"
               f" | level {row.level:.2f}"
               f" | NaN {row.divergences:d}"
               f" | KL {row.KL:.5f}"
               f" | lr {row.learning_rate:.2e}"
               f" | speed {row.max_speed:.2f}"
               f" | steps/s {row.steps_per_second:.0f}")
    print(message, flush=True)
