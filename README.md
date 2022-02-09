This should just be the same as PySoap but the main change is that we no longer assume the network is a Sequential model. Instead, we now assume that the model can take a tree like structure with the help of Split and Join layers.

Note that Dense, Conv, etc. are not assumed to have 2 children, because I'm not sure how the gradients would work. Instead the tree like structure comes from the fact that split takes 1 parent, and returns 2 childrens. With Join taking 2 parents, returning 1 children

To use the GPU implementation, all that is needed is PyOpenCl