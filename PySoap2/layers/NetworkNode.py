class NetworkNode(object):
    def __init__(self):
        self.parents = ()
        self.children = ()

    def add_parent(self, parent):
        """ Add parent node

            Parameters
            ---------
            parent : :obj:
        """
        self.parents += (parent,)

    def add_parents(self, parents):
        """ Add multiple parent nodes

            Parameters
            ---------
            parents : tuple of :obj:
        """

        self.parents += parents

    def add_child(self, child):
        """ Add child node

            Parameters
            ---------
            child : :obj:
        """
        self.children += (child,)

    def add_children(self, children):
        """ Add multiple child nodes

            Parameters
            ---------
            children : tuple of :obj:
        """
        self.children += children

    def __call__(self, parent_node):
        """ Connects the current node instance with a single parent node

            Parameters
            ----------
            parent_node : :obj:

            Raises
            ------
            ValueError
                If `parent_node` is a tuple, or
                If it is not an instance of NetworkNode
         """
        if isinstance(parent_node, tuple):
            raise ValueError('Call assumes a single object as the parent. '
                             'Overload __call__ if you wish to initialise with tuple')

        if not isinstance(parent_node, NetworkNode):
            raise ValueError('Call argument must inherit from NetworkNode')

        self.add_parent(parent_node)
        parent_node.add_child(self)

        return self
