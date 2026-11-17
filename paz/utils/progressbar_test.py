import jax
import jax.numpy as jp

import paz.utils.progressbar as utils_progressbar


def test_start_returns_float():
    assert isinstance(utils_progressbar.start(), float)


def test_newline_writes_newline(capsys):
    utils_progressbar.newline()
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert captured.out == "\n"


def test_draw_includes_description(capsys):
    start_time = utils_progressbar.start()
    utils_progressbar.draw(1, 10, start_time, "test", 10)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "test" in captured.out


def test_print_bar_includes_description(capsys):
    start_time = utils_progressbar.start()
    utils_progressbar.print_bar(1, 10, start_time, "test", 10)
    captured = capsys.readouterr()
    assert "test" in captured.out


def test_print_bar_appends_suffix(capsys):
    start_time = utils_progressbar.start()
    utils_progressbar.print_bar(
        1, 10, start_time, "test", 10, suffix="stop=loss"
    )
    captured = capsys.readouterr()
    assert captured.out.endswith(" | stop=loss")


def test_print_bar_clears_previous_tail():
    message = "\rshort"
    padded = utils_progressbar._pad_message(message)
    padded = utils_progressbar._pad_message("\rlonger message")
    padded = utils_progressbar._pad_message(message)
    assert padded.endswith(" " * (len("\rlonger message") - len(message)))


def test_draw_runs_under_jit(capsys):
    start_time = utils_progressbar.start()
    draw = jax.jit(
        lambda step: utils_progressbar.draw(
            step, 4, start_time, "jit", width=4
        )
    )
    draw(2)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "jit" in captured.out


def test_show_builds_callback_for_scan(capsys):
    callback = utils_progressbar.show(3, "scan", width=3)

    def body(carry, step):
        callback(step)
        return carry, None

    jax.jit(lambda: jax.lax.scan(body, None, jp.arange(3)))()
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert "scan" in captured.out


def show_two_stages(capsys, counts):
    stages = [("tune", 2, 4), ("sample", 4, 4)]
    reports = utils_progressbar.show_stages(stages)
    for index, count in counts:
        reports[index](count)
    jax.effects_barrier()
    output = capsys.readouterr().out
    return output, output.split("\r")[-1].rstrip()


def test_show_stages_draws_waiting_stage_empty(capsys):
    _, last_frame = show_two_stages(capsys, [(0, 1)])
    assert last_frame.startswith("tune |██--| 1/2 50% [")
    assert last_frame.endswith("sample |----| 0/4")


def test_show_stages_keeps_finished_stage_with_its_time(capsys):
    _, last_frame = show_two_stages(capsys, [(0, 2), (1, 2)])
    tune, sample = last_frame.split("  ")
    assert tune.startswith("tune |████| 2/2 ") and tune.endswith("s")
    assert sample.startswith("sample |██--| 2/4 50% [")


def test_show_stages_stays_on_one_line(capsys):
    output, _ = show_two_stages(capsys, [(0, 1), (0, 2), (1, 4)])
    assert "\n" not in output


def test_skip_returns_none():
    assert utils_progressbar.skip(3) is None
