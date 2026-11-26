import importlib.util
import os
import types
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_builder():
    path = ROOT / "docs" / "build.py"
    spec = importlib.util.spec_from_file_location("docs_build", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def scale(boxes, factor=2.0):
    """Scales boxes by a factor.

    # Arguments
    - `boxes`: Array of shape `(num_boxes, 4)`.

    # Example
    ```python
    # doubles every coordinate
    scale(boxes)
    ```
    """
    return boxes * factor


def translate(boxes, offset):
    return boxes + offset


def _clip(boxes):
    """Clips boxes."""
    return boxes


@pytest.fixture
def builder():
    return load_builder()


@pytest.fixture
def module():
    module = types.ModuleType("fake.boxes")
    for function in (translate, scale, _clip):
        function.__module__ = "fake.boxes"
        setattr(module, function.__name__, function)
    module.join = os.path.join
    return module


def test_collect_members_keeps_public_local_members_in_order(builder, module):
    names = [name for name, member in builder.collect_members(module)]
    assert names == ["translate", "scale"]


def test_select_documented_skips_members_without_docstring(builder, module):
    members = builder.collect_members(module)
    names = [name for name, member in builder.select_documented(members)]
    assert names == ["scale"]


def test_build_page_writes_documented_entries(builder, module):
    page = builder.build_page(module, ROOT)
    assert page.startswith("# fake.boxes")
    assert "## scale" in page
    assert "scale(boxes, factor=2.0)" in page
    assert "Scales boxes by a factor." in page
    assert "translate" not in page
    assert "_clip" not in page
    assert "join" not in page


def test_build_page_links_to_source_line(builder, module):
    page = builder.build_page(module, ROOT)
    line = scale.__code__.co_firstlineno
    url = "https://github.com/oarriaga/paz/blob/master"
    assert f"{url}/tests/docs/build_test.py#L{line}" in page


def test_build_page_demotes_docstring_headings(builder, module):
    lines = builder.build_page(module, ROOT).split("\n")
    assert "### Arguments" in lines
    assert "### Example" in lines
    assert "# Arguments" not in lines


def test_demote_headings_keeps_comments_in_code_blocks(builder):
    docstring = "# Example\n```python\n# a comment\n```\n# Returns"
    demoted = builder.demote_headings(docstring).split("\n")
    assert demoted[0] == "### Example"
    assert demoted[2] == "# a comment"
    assert demoted[4] == "### Returns"


def test_is_documented_ignores_generated_namedtuple_docstring(builder):
    Box = namedtuple("Box", ["x_min", "y_min"])
    assert not builder.is_documented(Box)


def test_build_page_path_mirrors_module_name(builder):
    path = builder.build_page_path("paz.backend.boxes")
    assert path == Path("api", "backend", "boxes.md")
