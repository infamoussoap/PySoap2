import numpy as np

from PySoap2.optimizers import Optimizer
from .LearningRateSchedulers import convert_to_learning_rate_scheduler


class Adadelta(Optimizer):
    """ Adam optimiser

        Attributes
        ----------
        learning_rate : float
            The learning rate or step size to take in the gradient given by adam
        rho : float
            Decay weight for moving average
        e : float
            Arbitrarily small float to prevent division by zero error

        dx2 : dict of int - np.array
            Stores the weighted average of (dx)^2
        v : dict of int - np.array
            Stores the weighted average second raw momentum
    """

    def __init__(self, learning_rate=1, rho=0.95, e=1e-7):
        """ Initialise attributes of Adam Optimiser

            Parameters
            ----------
            learning_rate : float, optional
            rho : float, optional
            e : float, optional
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.e = e

        self.learning_rate_scheduler = convert_to_learning_rate_scheduler(learning_rate)

        self.dx2 = None
        self.v = None
        self.t = 0

    def step(self, grad_dict):
        """ Returns the gradients as scheduled by Adadelta

            Parameters
            ----------
            grad_dict : dict of int - np.array
                Dictionary of gradients, where keys represent the layer number and the
                corresponding value is the layer gradients

            Returns
            -------
            dict of int - np.array
                Dictionary of the gradients as scheduled by Adadelta. The keys represent the
                layer number, and the corresponding value will be the scheduled gradient

            Notes
            -----
            This function returns the value to subtract from the current parameters.
            Consider grad_dict as dS/da, with a the parameters of the network. Then to
            update the parameters of the network

            a = a - Adadelta.gradients(dS/da)

            Obviously the output is a dictionary, so you'll have to account for that.
        """
        self.t += 1

        if self.t == 1:
            self.dx2 = {key: 0 for key in grad_dict.keys()}
            self.v = {key: 0 for key in grad_dict.keys()}

        # Compute 2nd moment of gradient for this iteration
        self.v = {key: self.rho * v + (1 - self.rho) * (g**2) if g is not None else None
                  for (key, v, g) in zip(grad_dict.keys(), self.v.values(), grad_dict.values())}

        # Compute step size for this iteration
        step = {key: np.sqrt(dx2 + self.e) * g / np.sqrt(v + self.e) if g is not None else None
                for (key, dx2, g, v) in zip(grad_dict.keys(), self.dx2.values(), grad_dict.values(), self.v.values())}

        # Compute dx2 for next iteration
        self.dx2 = {key: self.rho * dx2 + (1 - self.rho) * dx**2 if dx is not None else None
                    for (key, dx, dx2) in zip(step.keys(), step.values(), self.dx2.values())}

        scheduled_learning_rate = self.learning_rate_scheduler.get()
        return {key: scheduled_learning_rate * val if val is not None else None
                for (key, val) in step.items()}

    def new_instance(self):
        return Adadelta(self.learning_rate, self.rho, self.e)

    def parameters_(self):
        return {'dx2': self.dx2, 'v': self.v, 't': self.t}
