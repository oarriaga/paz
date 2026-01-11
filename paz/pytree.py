import json
import pickle
from pathlib import Path


class COLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_node_type_name(node):
    return f"{type(node).__name__}"


def is_container(node):
    is_dictionary_or_list = isinstance(node, (dict, list))
    is_namedtuple = isinstance(node, tuple) and hasattr(node, "_asdict")
    return is_dictionary_or_list or is_namedtuple


def print_leaf(node_name, node, prefix="", connector="└── "):
    title = f"{prefix}{connector}{COLORS.BOLD}{node_name}{COLORS.ENDC}"
    shape_str = f"{COLORS.OKGREEN}{node.shape}{COLORS.ENDC}"
    dtype_str = f"{COLORS.OKCYAN}{node.dtype}{COLORS.ENDC}"
    types = f"{shape_str} {dtype_str}"
    print(f"{title}: {types}")


def _traverse_and_print(node_name, node, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    if not is_container(node):  # It's a leaf, print its details
        if hasattr(node, "shape") and hasattr(node, "dtype"):
            print_leaf(node_name, node, prefix, connector)
        else:
            value_str = f"{COLORS.OKGREEN}{node}{COLORS.ENDC}"
            type_str = f"{COLORS.OKCYAN}{type(node).__name__}{COLORS.ENDC}"
            print(
                f"{prefix}{connector}{COLORS.BOLD}{node_name}{COLORS.ENDC}: {value_str} ({type_str})"
            )
        return

    # It's a container, print its name and recurse
    type_str = get_node_type_name(node)
    print(
        f"{prefix}{connector}{COLORS.BOLD}{node_name}{COLORS.ENDC} {type_str}"
    )

    new_prefix = prefix + ("    " if is_last else "│   ")

    # Get children from the container
    if hasattr(node, "_asdict"):
        children = list(node._asdict().items())
    elif isinstance(node, dict):
        children = list(node.items())
    else:  # list or simple tuple
        children = list(enumerate(node))

    num_children = len(children)
    for child_arg, (key, value) in enumerate(children):
        is_child_last = child_arg == num_children - 1
        _traverse_and_print(
            str(key), value, prefix=new_prefix, is_last=is_child_last
        )


def print(tree):
    type_name = get_node_type_name(tree)
    print(f"{COLORS.HEADER}{COLORS.BOLD}{type_name}{COLORS.ENDC}")

    if hasattr(tree, "_asdict"):
        children = list(tree._asdict().items())
    elif isinstance(tree, dict):
        children = list(tree.items())
    elif isinstance(tree, (list, tuple)):
        children = list(enumerate(tree))
    else:
        print("└── (Leaf Node)")
        _traverse_and_print(type_name, tree)
        return

    num_children = len(children)
    for child_arg, (key, value) in enumerate(children):
        is_child_last = child_arg == num_children - 1
        _traverse_and_print(str(key), value, prefix="", is_last=is_child_last)


def to_pickle(tree, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as filedata:
        pickle.dump(tree, filedata)


def to_json(tree, filepath, indent=4):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as filedata:
        json.dump(_to_serializable(tree), filedata, indent=indent)


def _to_serializable(tree):
    if hasattr(tree, "_asdict"):
        return {
            key: _to_serializable(value) for key, value in tree._asdict().items()
        }
    if isinstance(tree, dict):
        return {key: _to_serializable(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [_to_serializable(value) for value in tree]
    if hasattr(tree, "tolist"):
        return tree.tolist()
    return tree
