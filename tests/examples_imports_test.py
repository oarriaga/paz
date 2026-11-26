"""Checks that example scripts import names their own packages define.

Examples run work at module scope, so importing them here would execute
training loops and allocate GPUs. This walks them statically instead: for
every `from <local module> import <name>`, confirm the target file exists
and defines that name. That is the class of breakage a library change
causes when it removes a helper an example still imports, which no other
test catches because nothing imports the examples.

The same walk resolves every `paz.<name>` chain and `from paz... import`
against the installed package, so a renamed or removed library function
fails here instead of at the first run of the example.
"""
import ast
import importlib
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Examples that use paz names the library no longer has. Each is fixed on
# its own branch; delete the entry when the fix lands (strict xfail fails
# the run when a listed file starts passing).
KNOWN_BROKEN = {
    "object_detection/compute_MAP.py",
    "object_detection/demo_image.py",
    "object_detection/efficientdet.py",
    "pyramid_detector/detector.py",
    "pyramid_detector/pyramid.py",
}


def find_scripts():
    return sorted(EXAMPLES.rglob("*.py"))


def parse(path):
    return ast.parse(path.read_text(), filename=str(path))


def collect_defined_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(unpack_target_names(node.targets))
        elif isinstance(node, ast.AnnAssign):
            names.update(unpack_target_names([node.target]))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(unpack_import_names(node))
    return names


def unpack_target_names(targets):
    names = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(unpack_target_names(target.elts))
    return names


def unpack_import_names(node):
    names = set()
    for alias in node.names:
        names.add(alias.asname or alias.name.split(".")[0])
    return names


def resolve_local_module(script, node):
    """Return the file a local `from ... import ...` refers to, or None."""
    package = script.parent
    if node.level:
        for _ in range(node.level - 1):
            package = package.parent
    elif node.module is None:
        return None
    parts = node.module.split(".") if node.module else []
    if not node.level and parts and not (package / parts[0]).exists():
        return None
    candidate = package.joinpath(*parts)
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def collect_unresolved(script):
    tree = parse(script)
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = resolve_local_module(script, node)
        if module is None:
            continue
        defined = collect_defined_names(parse(module))
        for alias in node.names:
            if alias.name == "*" or alias.name in defined:
                continue
            if is_submodule(module, alias.name):
                continue
            missing.append(f"{node.module}.{alias.name}")
    return missing


def is_submodule(module, name):
    """`from package import module` imports a file, not a defined name."""
    directory = module.parent
    if (directory / f"{name}.py").is_file():
        return True
    return (directory / name / "__init__.py").is_file()


@pytest.mark.parametrize("script", find_scripts(), ids=lambda p: p.name)
def test_example_local_imports_resolve(script):
    missing = collect_unresolved(script)
    assert missing == [], f"{script} imports missing names: {missing}"


def mark_known_broken(script):
    name = script.relative_to(EXAMPLES).as_posix()
    marks = [pytest.mark.xfail(strict=True)] if name in KNOWN_BROKEN else []
    return pytest.param(script, marks=marks, id=name)


def collect_unresolved_paz_names(script):
    unresolved = []
    for name, line in collect_dotted_names(parse(script)):
        if is_paz_name(name) and not is_resolvable(name):
            unresolved.append(f"{script.relative_to(EXAMPLES)}:{line} {name}")
    return unresolved


def collect_dotted_names(tree):
    """Return `(dotted name, line)` for every import and name chain."""
    inner = collect_inner_attributes(tree)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.extend(unpack_import_from(node))
        elif isinstance(node, ast.Import):
            names.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.Attribute) and id(node) not in inner:
            names.append((unpack_attribute_chain(node), node.lineno))
    return sorted(set(names))


def collect_inner_attributes(tree):
    """Ids of attributes that are a prefix of a longer chain."""
    inner = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            inner.add(id(node.value))
    return inner


def unpack_import_from(node):
    names = []
    if node.level == 0:
        for alias in node.names:
            names.append((f"{node.module}.{alias.name}", node.lineno))
    return names


def unpack_attribute_chain(node):
    """Return `a.b.c` for a pure name chain, or an empty string."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    root = node.id if isinstance(node, ast.Name) else ""
    return ".".join([root, *reversed(parts)])


def is_paz_name(name):
    return name.startswith("paz.") and not name.endswith(".*")


def is_resolvable(name):
    try:
        load(name)
        resolvable = True
    except AttributeError:
        resolvable = False
    return resolvable


def load(name):
    """Return the module or attribute a dotted paz name refers to."""
    parent, _, attribute = name.rpartition(".")
    try:
        value = importlib.import_module(name)
    except ModuleNotFoundError as error:
        # A missing third-party module is not a broken paz name.
        if not name.startswith(error.name):
            raise
        value = getattr(load(parent), attribute)
    return value


@pytest.mark.parametrize("script", map(mark_known_broken, find_scripts()))
def test_example_paz_names_resolve(script):
    try:
        unresolved = collect_unresolved_paz_names(script)
    except ModuleNotFoundError as error:
        pytest.skip(f"optional dependency not installed: {error.name}")
    assert unresolved == [], f"unresolved paz names: {unresolved}"


def test_scripts_are_discovered():
    assert len(find_scripts()) > 50
