import warnings
from PySoap2.layers import NetworkNode as CPUNetworkNode


class GPUNetworkNode(object):
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
        if isinstance(parent_node, GPUNetworkNode):
            self.add_parent(parent_node)
            parent_node.add_child(self)
            return self
        elif isinstance(parent_node, (tuple, list)):
            raise ValueError('Call assumes a single object as the parent. '
                             'Overload __call__ if you wish to initialise with tuple')
        elif isinstance(parent_node, CPUNetworkNode):
            raise ValueError(f'Call argument is an instance of a CPU layer, but {type(self).__name__} '
                             'is an instance of GPU layer.')
        else:
            raise ValueError('Call argument must inherit from NetworkNode')


class SingleParentNetworkNode(GPUNetworkNode):
    """ Instance of `NetworkNode` but this node can only have 1 parent, with unlimited amount of children """
    def __init__(self):
        GPUNetworkNode.__init__(self)

    def add_parent(self, parent):
        # Only update parents if none exists
        if len(self.parents) == 0:
            self.parents = (parent,)
            return

        # Warn if the class already has a parent
        warnings.warn(f'{type(self).__name__} is an instance of SingleParentNetworkNode, meaning it can only '
                      'have 1 parent. But one parent was already given, and the new parent will be ignored.')

    def add_parents(self, parents):
        """ parents : tuple[NetworkNode] or list[NetworkNode] """
        if len(parents) > 1:
            warnings.warn(f'{type(self).__name__} is an instance of SingleParentNetworkNode, meaning it can only '
                          'have 1 parent. But more than 1 parent was given, and only the first parent will be'
                          'considered.')

        # Only update parents if none exists
        if len(self.parents) == 0:
            self.parents = (parents[0],)
            return

        # Warn if the class already has a parent
        warnings.warn(f'{type(self).__name__} is an instance of SingleParentNetworkNode, meaning it can only '
                      'have 1 parent. But one parent was already given, and the new parent will be ignored.')

    def does_parent_exists(self):
        return len(self.parents) > 0
