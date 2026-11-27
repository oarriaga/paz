from paz.backend import message


def test_info_prints_cyan_message(capsys):
    test_message = "This is info"
    message.info(test_message)

    captured = capsys.readouterr()
    assert "This is info" in captured.out
    assert "\033[96m" in captured.out
    assert "\033[0m" in captured.out


def test_success_prints_green_message(capsys):
    test_message = "This is success"
    message.success(test_message)

    captured = capsys.readouterr()
    assert "This is success" in captured.out
    assert "\033[92m" in captured.out
    assert "\033[0m" in captured.out


def test_error_prints_red_message(capsys):
    test_message = "This is an error"
    message.error(test_message)

    captured = capsys.readouterr()
    assert "This is an error" in captured.out
    assert "\033[91m" in captured.out
    assert "\033[0m" in captured.out


def test_warn_prints_yellow_message(capsys):
    test_message = "This is a warning"
    message.warn(test_message)

    captured = capsys.readouterr()
    assert "This is a warning" in captured.out
    assert "\033[93m" in captured.out
    assert "\033[0m" in captured.out


def test_debug_prints_gray_message(capsys):
    test_message = "This is debug"
    message.debug(test_message)

    captured = capsys.readouterr()
    assert "This is debug" in captured.out
    assert "\033[90m" in captured.out
    assert "\033[0m" in captured.out


def test_info_with_multiple_arguments(capsys):
    message.info("value:", 42, "percent:", 0.5)

    captured = capsys.readouterr()
    assert "value: 42 percent: 0.5" in captured.out


def test_warn_with_multiple_arguments(capsys):
    message.warn("sparsity=", 0.95, "magnitude=", 0.001)

    captured = capsys.readouterr()
    assert "sparsity= 0.95 magnitude= 0.001" in captured.out


def test_error_with_mixed_types(capsys):
    message.error("count:", 10, "items:", ["a", "b"])

    captured = capsys.readouterr()
    assert "count: 10 items: ['a', 'b']" in captured.out
