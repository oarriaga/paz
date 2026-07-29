"""Checks that example scripts import names their own packages define.

Examples run work at module scope, so importing them here would execute
training loops and allocate GPUs. This walks them statically instead: for
every `from <local module> import <name>`, confirm the target file exists
and defines that name. That is the class of breakage a library change
causes when it removes a helper an example still imports, which no other
test catches because nothing imports the examples.
"""
import ast
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def find_scripts():
    return sorted(path for path in EXAMPLES.rglob("*.py")
                  if "legacy" not in path.parts)


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


def test_scripts_are_discovered():
    assert len(find_scripts()) > 50
