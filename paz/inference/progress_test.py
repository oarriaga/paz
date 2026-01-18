import time

from paz.inference import progress


def test_now_returns_float():
    assert isinstance(progress.now(), float)


def test_move_to_next_line_writes_newline(capsys):
    progress.move_to_next_line()
    captured = capsys.readouterr()
    assert captured.out == "\n"


def test_draw_bar_includes_description(capsys):
    start_time = time.perf_counter() - 1.0
    progress.draw_bar(1, 10, start_time, "test", 10)
    captured = capsys.readouterr()
    assert "test" in captured.out


def test_build_bar_callback_writes_description(capsys):
    start_time = time.perf_counter() - 1.0
    callback = progress.build_bar_callback(2, start_time, "hello", 10)
    callback(1)
    captured = capsys.readouterr()
    assert "hello" in captured.out
