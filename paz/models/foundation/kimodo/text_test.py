import pytest

from paz.models.foundation.kimodo.text import sanitize_text

# Expected values come from running the reference sanitize_text.
CASES = [
    (" A person is    walking.", "A person is walking."),
    ("someone go forward", "Someone go forward."),
    ("jump", "Jump."),
    ("jumping!", "Jumping!"),
    ("jumping)", "Jumping)."),
    ("-go", "Go."),
    ("blocasdji  -----", "Blocasdji."),
    ("", ""),
    ("walks. then jumps", "Walks. Then jumps."),
    ("A person walks forward.", "A person walks forward."),
    ("a person waves hello!! then sits down",
     "A person waves hello! ! Then sits down."),
]


@pytest.mark.parametrize("text, expected", CASES)
def test_sanitize_text(text, expected):
    assert sanitize_text(text) == expected
