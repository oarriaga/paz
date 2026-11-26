import importlib
import inspect
import shutil
from pathlib import Path

MODULES = [
    "paz.backend.image",
    "paz.backend.boxes",
    "paz.backend.detection",
    "paz.backend.draw",
    "paz.backend.lie.SO3",
    "paz.backend.lie.SE3",
    "paz.graphics.renderer",
    "paz.graphics.camera",
    "paz.graphics.scene",
    "paz.graphics.shapes",
    "paz.inference.infer",
    "paz.inference.metropolis_hastings",
    "paz.optimization.minimization",
    "paz.optimization.stopping",
    "paz.losses.standard",
    "paz.losses.mesh",
    "paz.applications.detectors",
    "paz.applications.segmenters",
    "paz.abstract.sequential",
    "paz.abstract.hash_table",
    "paz.datasets.voc",
    "paz.datasets.cityscapes",
]


def build_page(module, root):
    entries = []
    for name, member in select_documented(collect_members(module)):
        entries.append(build_entry(name, member, root))
    return "\n\n".join([f"# {module.__name__}", *entries]) + "\n"


def build_page_path(module_name):
    parts = module_name.split(".")[1:]
    return Path("api", *parts[:-1], parts[-1] + ".md")


def collect_members(module):
    members = []
    for name, member in vars(module).items():
        if is_public_member(module, name, member):
            members.append((name, member))
    return members


def select_documented(members):
    return [(name, member) for name, member in members if is_documented(member)]


def is_documented(member):
    # namedtuple writes a docstring of its own listing the field names.
    fields = ", ".join(getattr(member, "_fields", ()))
    generated = f"{member.__name__}({fields})"
    return bool(member.__doc__) and member.__doc__ != generated


def is_public_member(module, name, member):
    is_public = not name.startswith("_")
    is_local = getattr(member, "__module__", None) == module.__name__
    return is_public and is_local and is_function_or_class(member)


def is_function_or_class(member):
    is_function = inspect.isfunction(inspect.unwrap(member))
    return is_function or inspect.isclass(member)


def build_entry(name, member, root):
    signature = f"```python\n{name}{inspect.signature(member)}\n```"
    source = f"[[source]]({build_source_link(member, root)})"
    docstring = demote_headings(inspect.cleandoc(member.__doc__))
    return "\n\n".join([f"## {name}", signature, source, docstring])


def build_source_link(member, root):
    member = inspect.unwrap(member)
    path = Path(inspect.getsourcefile(member)).resolve().relative_to(root)
    line = inspect.getsourcelines(member)[1]
    url = "https://github.com/oarriaga/paz/blob/master"
    return f"{url}/{path.as_posix()}#L{line}"


def demote_headings(docstring):
    # A "#" inside a fenced code block is a comment, not a heading.
    lines, is_code = [], False
    for line in docstring.split("\n"):
        is_code = is_code != line.lstrip().startswith("```")
        is_heading = line.startswith("#") and not is_code
        lines.append("##" + line if is_heading else line)
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "docs" / "sources"
    shutil.rmtree(output, ignore_errors=True)
    shutil.copytree(root / "docs" / "guides", output)
    num_documented, num_total = 0, 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        members = collect_members(module)
        documented = select_documented(members)
        page_path = output / build_page_path(module_name)
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(build_page(module, root))
        print(f"{module_name}: {len(documented)}/{len(members)}")
        num_documented = num_documented + len(documented)
        num_total = num_total + len(members)
    print(f"total: {num_documented}/{num_total}")
