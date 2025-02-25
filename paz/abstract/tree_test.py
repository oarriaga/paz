import pytest

from paz.abstract.tree import Tree, NO_PARENT


@pytest.fixture
def chain():
    tree = Tree(["d", "c", "b", "a"])
    tree.add_edge("a", "b")
    tree.add_edge("b", "c")
    tree.add_edge("c", "d")
    return tree


@pytest.fixture
def chain_parent_array():
    return [NO_PARENT, 0, 1, 2]


@pytest.fixture
def chain_leaves():
    return ["d"]


@pytest.fixture
def chain_root():
    return ["a"]


@pytest.fixture
def chain_sorted_nodes():
    return ["a", "b", "c", "d"]


@pytest.fixture
def fork():
    tree = Tree(["a", "b", "c", "d"])
    tree.add_edge("a", "b")
    tree.add_edge("b", "c")
    tree.add_edge("b", "d")
    return tree


@pytest.fixture
def fork_root():
    return ["a"]


@pytest.fixture
def fork_leaves():
    return ["c", "d"]


@pytest.fixture
def fork_parent_array():
    return [NO_PARENT, 0, 1, 1]


@pytest.fixture
def fork_sorted_nodes():
    return ["a", "b", "c", "d"]


@pytest.fixture
def collider():
    tree = Tree(["a", "b", "c"])
    tree.add_edge("a", "c")
    tree.add_edge("b", "c")
    return tree


def test_chain_root(chain, chain_root):
    assert chain.root_nodes() == chain_root


def test_chain_sort(chain, chain_sorted_nodes):
    assert chain.sort_topologically() == chain_sorted_nodes


def test_chain_connection(chain):
    assert chain.is_weakly_connected()


def test_chain_parent_array(chain, chain_parent_array):
    sorted_nodes = chain.sort_topologically()
    parent_array = chain.parent_array(sorted_nodes)
    assert parent_array == chain_parent_array


def test_chain_leaves(chain, chain_leaves):
    assert chain.leaves() == chain_leaves


def test_fork_root(fork, fork_root):
    assert fork.root_nodes() == fork_root


def test_fork_sort(fork, fork_sorted_nodes):
    assert fork.sort_topologically() == fork_sorted_nodes


def test_fork_connection(fork):
    assert fork.is_weakly_connected()


def test_fork_parent_array(fork, fork_parent_array):
    sorted_nodes = fork.sort_topologically()
    parent_array = fork.parent_array(sorted_nodes)
    assert parent_array == fork_parent_array


def test_fork_leaves(fork, fork_leaves):
    assert fork.leaves() == fork_leaves


def test_no_colliders():
    tree = Tree(["a", "b", "c"])
    tree.add_edge("a", "c")
    pytest.raises(ValueError, tree.add_edge, "b", "c")


def test_no_repeated_nodes():
    tree = Tree(["a", "b", "c"])
    pytest.raises(ValueError, tree.add_node, "a")


def test_no_repeated_edges():
    tree = Tree(["a", "b", "c"])
    tree.add_edge("a", "b")
    tree.add_edge("b", "c")
    pytest.raises(ValueError, tree.add_edge, "b", "c")


def test_no_source_edge():
    tree = Tree(["a", "b", "c"])
    pytest.raises(ValueError, tree.add_edge, "d", "a")


def test_no_target_edge():
    tree = Tree(["a", "b", "c"])
    pytest.raises(ValueError, tree.add_edge, "a", "d")


def test_node_constructor():
    tree_A = Tree(["a", "b", "c"])
    tree_B = Tree()
    tree_B.add_node("a")
    tree_B.add_node("b")
    tree_B.add_node("c")
    assert tree_A.nodes == tree_B.nodes
    assert tree_A.sort_topologically() == tree_B.sort_topologically()
