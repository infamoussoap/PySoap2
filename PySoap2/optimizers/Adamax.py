import numpy as np

from PySoap2.optimizers import Optimizer


class Adamax(Optimizer):
    """ Adamax optimiser

        Attributes
        ----------
        learning_rate : float
            The learning rate or step size to take in the gradient given by adam
        b1 : float
            Decay rate for first momentum. Must be 0 < b1 < 1
        b2 : float
            Decay rate for exponentially weighted infinity norm. Must be 0 < b2 < 1
        e : float
            Arbitrarily small float to prevent division by zero error

        t : int
            The time step, or the number of times the instance of Adam was called
        m : dict of int - np.array
            Stores the previous value of the first momentum
        v : dict of int - np.array
            Stores the previous value of the exponentially weighted infinity norm
    """

    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-7):
        """ Initialise attributes of Adamax Optimiser

            Parameters
            ----------
            learning_rate : float, optional
            b1 : float, optional
            b2 : float, optional
            e : float, optional
        """
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.e = e

        self.m = None
        self.v = None
        self.t = 0

    def step(self, grad_dict):
        """ Returns the gradients as scheduled by Adamax

            Parameters
            ----------
            grad_dict : dict of int - np.array
                Dictionary of gradients, where keys represent the layer number and the
                corresponding value is the layer gradients

            Returns
            -------
            dict of int - np.array
                Dictionary of the gradients as scheduled by Adamax. The keys represent the
                layer number, and the corresponding value will be the scheduled gradient

            Notes
            -----
            This function returns the value to subtract from the current parameters.
            Consider grad_dict as dS/da, with a the parameters of the network. Then to
            update the parameters of the network

            a = a - Adamax.gradients(dS/da)

            Obviously the output is a dictionary, so you'll have to account for that.
        """
        if self.t == 0:
            self.m = {key: 0 for key in grad_dict.keys()}
            self.v = {key: 0 for key in grad_dict.keys()}

        self.t += 1

        self.m = {key: self.b1 * m + (1 - self.b1) * g if g is not None else None
                  for (key, m, g) in zip(grad_dict.keys(), self.m.values(), grad_dict.values())}
        self.v = {key: np.maximum(self.b2 * v, np.abs(g)) if g is not None else None
                  for (key, v, g) in zip(grad_dict.keys(), self.v.values(), grad_dict.values())}

        a = self.learning_rate / (1 - self.b1 ** self.t)

        return {key: a * m / (v + self.e) if v is not None else None
                for (key, m, v) in zip(self.m.keys(), self.m.values(), self.v.values())}

    def new_instance(self):
        return Adamax(self.learning_rate, self.b1, self.b2, self.e)

    def parameters_(self):
        return {'m': self.m, 'v': self.v, 't': self.t}
