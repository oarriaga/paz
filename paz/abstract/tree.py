NO_PARENT = -1


class Tree:
    def __init__(self, nodes=None, edges=None, name=None):
        self.clear()
        if nodes is not None:
            [self.add_node(node) for node in nodes]
        if edges is not None:
            [self.add_edge(source, target) for (source, target) in edges]
        self.name = name

    def clear(self):
        self._nodes = []
        self._edges = []
        self._parent = {}
        self._children = {}

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if (isinstance(value, str)) or (value is None):
            self._name = value
        else:
            raise ValueError("Value must be a string or None")

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @property
    def is_acyclic_graph(self):
        return True  # by construction there are no cycles

    def parent(self, node):
        return self._parent[node]

    def children(self, node):
        return self._children[node]

    def sort_topologically(self):
        """Sort topologically tree with breadth-first search."""
        sorted_nodes, queue = [], self.root_nodes().copy()
        while len(queue) != 0:
            node = queue.pop(0)
            queue.extend(self.children(node))
            parent = self.parent(node)
            if (parent == NO_PARENT) or (parent in sorted_nodes):
                sorted_nodes.append(node)
        # return sorted_nodes
        return list(set(sorted_nodes))

    def parent_array(self, sorted_nodes):
        parent_array = []
        for node in sorted_nodes:
            parent = self.parent(node)
            if parent == NO_PARENT:
                parent_array.append(NO_PARENT)
            else:
                parent_arg = sorted_nodes.index(parent)
                parent_array.append(parent_arg)
        return parent_array

    def root_nodes(self):
        root_nodes = []
        for node in self.nodes:
            if self.parent(node) == NO_PARENT:
                root_nodes.append(node)
        return root_nodes

    def leaves(self):
        leaves = []
        for node in self.nodes:
            num_successors = len(self.children(node))
            if num_successors == 0:
                leaves.append(node)
        return leaves

    def add_node(self, node):
        if node in self.nodes:
            raise ValueError(f"Node {node} already in graph")
        self._nodes.append(node)
        self._children[node] = []
        self._parent[node] = NO_PARENT

    def add_edge(self, source_node, target_node):
        if source_node not in self.nodes:
            raise ValueError(f"Source {source_node} was not found in graph")
        if target_node not in self.nodes:
            raise ValueError(f"Target {target_node} was not found in graph")
        if {source_node: target_node} in self.edges:
            raise ValueError("Found edge already from source to target")
        if {target_node: source_node} in self.edges:
            raise ValueError("Found edge already from target to source: cycle")
        # if self._parent[target_node] != NO_PARENT:
        #     raise ValueError('Target {target_node} has already a parent')
        self._edges.append({source_node: target_node})
        self._children[source_node].append(target_node)
        self._parent[target_node] = source_node

    def is_weakly_connected(self):
        # full breadth-first search
        visited_nodes = []
        queue = self.root_nodes().copy()
        while len(queue) != 0:
            first_in = queue.pop(0)
            queue.extend(self.children(first_in))
            visited_nodes.append(first_in)
        difference_nodes = set(self.nodes).difference(set(visited_nodes))
        return True if len(difference_nodes) == 0 else False
