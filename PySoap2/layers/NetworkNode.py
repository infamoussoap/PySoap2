class NetworkNode(object):
    def __init__(self):
        self.parents = ()
        self.children = ()

    def add_parent(self, parent):
        self.parents += (parent,)

    def add_parents(self, parents):
        self.parents += parents

    def add_child(self, child):
        self.children += (child,)

    def add_children(self, children):
        self.children += children

    def __call__(self, parent_node):
        if not isinstance(parent_node, NetworkNode):
            raise ValueError('Call argument must be of type NetworkNode')

        self.add_parent(parent_node)
        parent_node.add_child(self)

        return self
