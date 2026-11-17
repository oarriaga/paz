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


def test_build_stage_draws_bar_and_count():
    stage = utils_progressbar.build_stage("tune", 5, 10, 10)
    assert stage == "tune |█████-----| 5/10"


def test_build_timing_shows_percent():
    start_time = utils_progressbar.start()
    assert utils_progressbar.build_timing(5, 10, start_time).startswith("50% [")


def test_show_text_rewrites_line_without_newline(capsys):
    show = utils_progressbar.show_text(lambda count: f"count {count}")
    show(3)
    jax.effects_barrier()
    captured = capsys.readouterr()
    assert captured.out.startswith("\rcount 3")
    assert "\n" not in captured.out
