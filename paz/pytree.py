from typing import Any


# Define a color class for prettier output in terminals that support it
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _traverse_and_print(
    node_name: str, node: Any, prefix: str = "", is_last: bool = True
):
    """
    Recursively traverses a PyTree node and prints its structure.
    Handles both array-like leaves and scalar leaves (int, float, etc.).
    """
    # Determine the connector for the tree structure
    connector = "└── " if is_last else "├── "

    # Check if the node is a leaf (not a dict, list, or tuple)
    if not isinstance(node, (dict, list, tuple)):
        # --- MODIFICATION START ---
        # Check if the leaf is array-like or a scalar
        if hasattr(node, "shape") and hasattr(node, "dtype"):
            # It's an array-like leaf, print its details
            shape_str = f"{bcolors.OKGREEN}{node.shape}{bcolors.ENDC}"
            dtype_str = f"{bcolors.OKCYAN}{node.dtype}{bcolors.ENDC}"
            print(
                f"{prefix}{connector}{bcolors.BOLD}{node_name}{bcolors.ENDC}: {shape_str} {dtype_str}"
            )
        else:
            # It's a scalar leaf (e.g., int, float), print its value and type
            value_str = f"{bcolors.OKGREEN}{node}{bcolors.ENDC}"
            type_str = f"{bcolors.OKCYAN}{type(node).__name__}{bcolors.ENDC}"
            print(
                f"{prefix}{connector}{bcolors.BOLD}{node_name}{bcolors.ENDC}: {value_str} ({type_str})"
            )
        # --- MODIFICATION END ---
        return

    # It's a container, print its name and type
    type_str = f"({type(node).__name__})"
    print(
        f"{prefix}{connector}{bcolors.BOLD}{node_name}{bcolors.ENDC} {type_str}"
    )

    # Prepare for recursion on children
    new_prefix = prefix + ("    " if is_last else "│   ")

    if isinstance(node, dict):
        children = list(node.items())
    else:  # list or tuple
        children = list(enumerate(node))

    num_children = len(children)
    for i, (key, value) in enumerate(children):
        is_child_last = i == num_children - 1
        _traverse_and_print(
            str(key), value, prefix=new_prefix, is_last=is_child_last
        )


def print(tree: Any, name: str = "Pytree"):
    """
    Prints the shape and structure of a JAX PyTree in a user-friendly format.

    Args:
        tree (Any): The PyTree (e.g., nested dicts/lists of JAX arrays) to print.
        name (str): The root name for the PyTree.
    """
    print(
        f"{bcolors.HEADER}{bcolors.BOLD}{name}{bcolors.ENDC} ({type(tree).__name__})"
    )

    # Start the traversal
    if isinstance(tree, dict):
        children = list(tree.items())
    elif isinstance(tree, (list, tuple)):
        children = list(enumerate(tree))
    else:  # It's a single leaf
        _traverse_and_print(name, tree)
        return

    num_children = len(children)
    for i, (key, value) in enumerate(children):
        is_child_last = i == num_children - 1
        _traverse_and_print(str(key), value, prefix="", is_last=is_child_last)
