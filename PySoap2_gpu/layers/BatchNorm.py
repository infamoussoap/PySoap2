import numpy as np
from functools import reduce

import pyopencl.array as cl_array
from pyopencl import clmath

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from PySoap2_gpu.utils import ClArrayTricks
from PySoap2_gpu.utils import Broadcast

from .ValueChecks import check_built


class BatchNormGrads:
    """ See the BatchNormGrads in PySoap2 for documentation """
    device_context = None
    device_queue = None

    initialized = False

    def __init__(self, context, queue):
        BatchNormGrads.device_context = context
        BatchNormGrads.device_queue = queue

        if not ClArrayTricks.initialized:
            ClArrayTricks(context, queue)

        if not Broadcast.initialized:
            Broadcast(context, queue)

        ClArrayTricks.initialized = True

    @staticmethod
    def dgamma(new_delta, z_hat):
        """
            Parameters
            ----------
            new_delta : (N, ...) cl_array.Array
            z_hat : (N, ...) cl_array.Array
        """

        return ClArrayTricks.sum_across_0_axis(new_delta * z_hat)

    @staticmethod
    def dbeta(new_delta):
        return ClArrayTricks.sum_across_0_axis(new_delta)

    @staticmethod
    def dz_hat(new_delta, gamma):
        return Broadcast.broadcast_across_0_axis("*", new_delta, gamma)

    @staticmethod
    def dsigma2(z, dz_hat_, epsilon, mu=None, sigma=None):
        """ epsilon is assumed to be a float, not a np.array """
        if mu is None:
            mu = ClArrayTricks.mean_across_0_axis(z)
        if sigma is None:
            sigma = ClArrayTricks.std_across_0_axis(z)

        c = (-0.5 * (sigma ** 2 + epsilon) ** (-3 / 2))

        mean_corrected_z = Broadcast.broadcast_across_0_axis("-", z, mu)

        return c * ClArrayTricks.sum_across_0_axis(dz_hat_ * mean_corrected_z)

    @staticmethod
    def dmu(z, dz_hat_, epsilon, sigma=None):
        """ epsilon is assumed to be a float, not a np.array """
        if sigma is None:
            sigma = ClArrayTricks.std_across_0_axis(z)

        sum_dz_hat_ = ClArrayTricks.sum_across_0_axis(dz_hat_)
        return (-1 / clmath.sqrt(sigma ** 2 + epsilon)) * sum_dz_hat_

    @staticmethod
    def dz(z, new_delta, gamma, epsilon, mu=None, sigma=None):
        """ epsilon is assumed to be a float, not a np.array """
        if mu is None:
            mu = ClArrayTricks.mean_across_0_axis(z)
        if sigma is None:
            sigma = ClArrayTricks.std_across_0_axis(z)
        m = len(z)

        dz_hat_ = BatchNormGrads.dz_hat(new_delta, gamma)
        dsigma2_ = BatchNormGrads.dsigma2(z, dz_hat_, epsilon, mu, sigma)
        dmu_ = BatchNormGrads.dmu(z, dz_hat_, epsilon, sigma)

        mean_corrected_z = Broadcast.broadcast_across_0_axis("-", z, mu)

        temp = Broadcast.broadcast_across_0_axis('/', dz_hat_, clmath.sqrt(sigma ** 2 + epsilon)) \
               + Broadcast.broadcast_across_0_axis("*", 2 * mean_corrected_z / m, dsigma2_)

        return Broadcast.broadcast_across_0_axis('+', temp, dmu_ / m)


class BatchNorm(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, device_context, device_queue):
        """ Initialise Attributes `gamma` and `beta` """
        self.device_context = device_context
        self.device_queue = device_queue

        if not BatchNormGrads.initialized:
            BatchNormGrads(device_context, device_queue)
        if not ClArrayTricks.initialized:
            ClArrayTricks(device_context, device_queue)

        self.input_shape = self.parents[0].output_shape  # It is assumed there is only one parent for this class
        self.output_shape = self.input_shape

        self.gamma = cl_array.zeros(device_queue, self.input_shape, np.float32) + 1
        self.beta = cl_array.zeros(device_queue, self.input_shape, np.float32)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, **kwargs):
        # The studentized z: (z - mu)/std
        z_hat = self.normalize(z, self.epsilon)

        # z_hat * gamma + beta
        out = Broadcast.broadcast_across_0_axis("+",
                                                Broadcast.broadcast_across_0_axis("*", z_hat, self.gamma),
                                                self.beta)

        if output_only:
            return out
        return out, out

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(lambda x, y: x + y, new_delta)

        dz_ = BatchNormGrads.dz(prev_z, delta, self.gamma, self.epsilon)
        return prev_z * dz_

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        delta = reduce(lambda x, y: x + y, delta)

        # The studentized z: (z - mu)/std
        z_hat = self.normalize(prev_z, self.epsilon)

        parameter_gradients = {'beta': ClArrayTricks.sum_across_0_axis(delta),
                               'gamma': ClArrayTricks.sum_across_0_axis(delta * z_hat)}
        return parameter_gradients

    @check_built
    def update_parameters_(self, parameter_updates):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            parameter_updates : dict of str - np.array
                The step size for the parameters, as scheduled by the optimizer
        """

        self.beta -= parameter_updates['beta']
        self.gamma -= parameter_updates['gamma']

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {'beta': self.beta.get(), 'gamma': self.gamma.get()}
        return self.beta.get(), self.gamma.get()

    @check_built
    def summary_(self):
        return f'Batch Norm', f"Output Shape {(None, *self.output_shape)}"

    def __str__(self):
        return f'Batch Norm; built = {self.built}'

    @staticmethod
    def normalize(z, epsilon=1e-7):
        mean = ClArrayTricks.mean_across_0_axis(z)
        std = ClArrayTricks.std_across_0_axis(z)

        # The studentized z: (z - mu)/std
        z_hat = Broadcast.broadcast_across_0_axis("/",
                                                  Broadcast.broadcast_across_0_axis("-", z, mean),
                                                  clmath.sqrt(std ** 2 + epsilon))

        return z_hat
