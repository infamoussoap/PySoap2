# import numpy as np
from functools import reduce

import pyopencl.array as cl_array
from pyopencl import clmath

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from PySoap2_gpu.utils import ClArrayTricks
from PySoap2_gpu.utils import Broadcast

from .Split import SplitInterfaceToDevice
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
        return new_delta * gamma

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
