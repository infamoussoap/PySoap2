import numpy as np
from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class BatchNormGrads:
    """ Returns the gradients (partial derivatives) of this layer.
        Note that d is meant to be partial

        1. dgamma = dS/d(gamma)
        2. dbeta = dS/d(beta)
        3. dz = dS/d(z^{k-1})
            This is the gradient with respect to the input of the batch-norm layer
        4. dz_hat = dS/d(z_hat^{k-1})
            The gradient with respect to the normalised input of the batch-norm layer
        5. dsigma2 = dS/d(sigma^2)
        6. dmu = dS/d(mu)
    """

    @staticmethod
    def dgamma(new_delta, z_hat):
        """ Returns ds/d(gamma)

            Parameters
            ----------
            new_delta : (N, ...) np.array
                Should be delta^{k}
            z_hat : (N, ...) np.array
                Should be the normalised input of this layer

            Returns
            -------
            (N, ...) np.array
        """
        # This can probably be speed up by using einsum, so that an intermediate np.array
        # isn't created
        return np.sum(new_delta * z_hat, axis=0)

    @staticmethod
    def dbeta(new_delta):
        """ Returns ds/d(beta)

            Parameters
            ----------
            new_delta : (N, ...) np.array
                Should be delta^{k}

            Returns
            -------
            (N, ...) np.array
        """

        return np.sum(new_delta, axis=0)

    @staticmethod
    def dz_hat(new_delta, gamma):
        """ Returns dS/d(z_hat) - The gradient with respect to the normalised input of
            the batch-norm layer

            Parameters
            ----------
            new_delta : (N, ...) np.array
                Should be delta^{k}
            gamma : (...) np.array

            Return
            ------
            (N, ...) np.array
        """
        return new_delta * gamma

    @staticmethod
    def dsigma2(z, dz_hat_, epsilon, mu=None, sigma=None):
        """ Returns dS/d(sigma^2)

            Parameters
            ----------
            z : (N, ...) np.array
                The input of this layer: z^{k-1}
            dz_hat_ : (N, ...) np.array
                The gradient with respect to the normalised input: dS/d(z_hat^{k-1})
            epsilon : float

            mu : (...) np.array, optional
                The mean of the input. If None, then it will be computed
            sigma : (...) np.array, optional
                The std of the input. If None, then it will be computed

            Returns
            -------
            (...) np.array
        """
        if mu is None:
            mu = np.mean(z, axis=0)
        if sigma is None:
            sigma = np.std(z, axis=0)

        c = (-0.5 * (sigma ** 2 + epsilon) ** (-3 / 2))
        return c * np.sum(dz_hat_ * (z - mu), axis=0)

    @staticmethod
    def dmu(z, dz_hat_, epsilon, mu=None, sigma=None, dsigma2_=None):
        """ Returns dS/dmu

            Parameters
            ----------
            z : (N, ...) np.array
                The input of this layer: z^{k-1}
            dz_hat_ : (N, ...) np.array
                The gradient with respect to the normalised input: dS/d(z_hat^{k-1})
            epsilon : float

            mu : (...) np.array, optional
                The mean of the input. If None, then it will be computed
            sigma : (...) np.array, optional
                The std of the input. If None, then it will be computed
            dsigma2_ : (...) np.array, optional
                This should be the gradient ds/d(sigma^2). If it is set to None then it
                will be computed when this function is called
        """

        if mu is None:
            mu = np.mean(z, axis=0)
        if sigma is None:
            sigma = np.std(z, axis=0)
        if dsigma2_ is None:
            dsigma2_ = BatchNormGrads.dsigma2(z, dz_hat_, epsilon, mu, sigma)

        return (-1 / np.sqrt(sigma**2 + epsilon))*np.sum(dz_hat_, axis=0) + dsigma2_ * np.mean(-2*(z - mu), axis=0)

    @staticmethod
    def dz(z, new_delta, gamma, epsilon, mu=None, sigma=None):
        """ Returns the partial derivative with respect to the input: dS/dZ^{n-1}

            Parameters
            ----------
            z : (N, ...) np.array
                The input of this layer: z^{n-1}
            new_delta : (N, ...) np.array
                The back-prop gradient: delta^{n}
            gamma : (...) np.array
            epsilon : float
                Arbitrarily small float to prevent division by 0 error

            mu : (...) np.array, optional
                The mean of the input. If None, then it will be computed
            sigma : (...) np.array, optional
                The std of the input. If None, then it will be computed

            Returns
            -------
            (N, ...) np.array
        """

        if mu is None:
            mu = np.mean(z, axis=0)
        if sigma is None:
            sigma = np.std(z, axis=0)
        m = len(z)

        dz_hat_ = BatchNormGrads.dz_hat(new_delta, gamma)
        dsigma2_ = BatchNormGrads.dsigma2(z, dz_hat_, epsilon, mu, sigma)
        dmu_ = BatchNormGrads.dmu(z, dz_hat_, epsilon, mu, sigma, dsigma2_)

        return dz_hat_ / np.sqrt(sigma**2 + epsilon) + dsigma2_*2*(z - mu)/m + dmu_/m


class BatchNorm(NetworkNode, LayerBaseAttributes, Layer):
    """ A BatchNorm layer where the inputs are normalised and then linearly scaled.
        Concretely, given an input z, this layer will return
            gamma * z_hat + beta
        where z_hat is the normalised version of z, and gamma and beta are matrices

        Attributes
        ----------
        input_shape : k tuple
            The shape of the input of this layer
        output_shape : k tuple
            The shape of the output of this layer
        gamma : np.array (of dimension k)
            The scaling factor of the elements
        beta : np.array (of dimension k)
            The bias units for the elements

        built : bool
            Has the model been initialised

        Notes
        -----
        This implementation of batch-norm assumes that batch-norm is applied AFTER the activation function. For
        example:
            Correct Use: Fully Connected -> Activation Function -> Batch-Norm (Batch-norm after activation)
            Incorrect Use: Fully Connected -> Batch-Norm -> Activation Function (Batch-norm before activation)
    """

    def __init__(self):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.epsilon = 1e-10
        self.gamma = None
        self.beta = None

    def build(self):
        """ Initialise Attributes `gamma` and `beta` """
        self.input_shape = self.parents[0].output_shape  # It is assumed there is only one parent for this class
        self.output_shape = self.input_shape

        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, **kwargs):
        """ Returns the output of this layer

            Parameters
            ----------
            z : (N, ...) np.array
                z is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z is the index that inputs is accessed by
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            (N, ...) np.array
                The final output of the layer, post activation

            OR (if `output_only = False`)

            (N, ...) np.array, (N, ...) np.array
                The first np.array will store the output before it is passed through the activation
                function.
                The second np.array will store the output after it has passed through the
                activation function.

            Notes
            -----
            Since the activation function is linear the 2 arrays, when output_only = True, are the same
            array
        """
        mean = np.mean(z, axis=0)
        std = np.std(z, axis=0)

        a = self.gamma * ((z - mean) / np.sqrt(std ** 2 + self.epsilon)) + self.beta

        if output_only:
            return a
        return a, a

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, ...) np.array
            new_delta : list of (N, ...) np.array
            prev_z : (N, ...) np.array

            Returns
            -------
            (N, ...) np.array
                Returns delta of the previous layer, delta^{k-1}

            Notes
            -----
            We want to return delta^{k-1} because the `sequential` class does not have access to the
            weights, W. But it does know the values of g'_{k-1} and delta^k, due to forward propagation
            and the backwards nature of the back propagation algorithm.
        """
        delta = np.sum(np.array(new_delta), axis=0)

        dz_ = BatchNormGrads.dz(prev_z, delta, self.gamma, self.epsilon)
        return dz_ * prev_z

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        """ Returns the gradients with respect to beta and gamma

            Parameters
            ----------
            delta : (N, ...) np.array
                Should be delta^k
            prev_z : (N, ...) np.array
                The input of this layer: z^{k-1}

            Returns
            -------
            dict of str - np.array
        """
        delta = reduce(lambda x, y: x + y, delta)

        z_hat = (prev_z - np.mean(prev_z, axis=0)) / np.sqrt(np.std(prev_z, axis=0) ** 2 + self.epsilon)

        parameter_gradients = {'beta': np.sum(delta, axis=0), 'gamma': np.sum(delta * z_hat, axis=0)}
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
    def get_weights(self):
        return self.beta, self.gamma

    @check_built
    def summary_(self):
        return f'Batch Norm', f"Output Shape {(None, *self.output_shape)}"

    def __str__(self):
        return f'Batch Norm; built = {self.built}'
