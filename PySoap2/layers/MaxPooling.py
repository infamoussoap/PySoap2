import numpy as np
from numpy.lib.stride_tricks import as_strided

from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class MaxPooling2D(NetworkNode, LayerBaseAttributes, Layer):
    @staticmethod
    def im2window(images, filter_spartial_shape, stride):
        """ Returns the sliding window result when given multiple z

            Parameters
            ----------
            images : (N, i, j, k) np.array
                Assumed to be many z, which is accessed by the 0th index, for which the
                window is to be slid across
            filter_spartial_shape : 2 tuple
                The spartial dimensions of the filter. Note that this can also simply be the filter
                size, assume that the first 2 dimensions are the spartial dimensions (which it should be)
            stride : int
                The stride of the filter

            Returns
            -------
            (N, i, j, k, l, m) np.array
                The results of the sliding window
        """

        image_num = images.shape[0]

        height = (images.shape[1] - filter_spartial_shape[0]) // stride + 1
        width = (images.shape[2] - filter_spartial_shape[1]) // stride + 1

        strides = (images.strides[0], images.strides[1] * stride, images.strides[2] * stride, *images.strides[1:])
        windowed = as_strided(images, (image_num, height, width, *filter_spartial_shape, images.shape[3]),
                              strides=strides)

        return windowed

    @staticmethod
    def im2col(images, filter_spartial_shape, stride):
        """ Returns the flatten sliding window result when given multiple z

            Parameters
            ----------
            images : (N, i, j, k) np.array
                Assumed to be many z, which is accessed by the 0th index, for which the
                window is to be slid across
            filter_spartial_shape : 2 tuple
                The spartial dimensions of the filter. Note that this can also simply be the filter
                size, assume that the first 2 dimensions are the spartial dimensions (which it should be)
            stride : int
                The stride of the filter

            Returns
            -------
            (N, i, j, k*l*m) np.array
                The results of the sliding window

            Notes
            -----
            The result of `im2col` and `im2window` are effectively the same thing. But `im2col` allows for
            matrix multiplication when performing the convolution, which is far faster than the alternative
            broadcasting
            Also note that we can't simply use `as_strided` here, and we need to reshape the results from
            `im2window`. The reason is that with `as_strided` there is no way to jump down a row once you
            reached the end of the filter spatial size. You can only do this by returning the windowed results
        """

        windowed = MaxPool2D.im2window(images, filter_spartial_shape, stride)
        shape = windowed.shape
        return windowed.reshape(shape[0], shape[1], shape[2], np.prod(shape[3:]))

    def __init__(self, pool_spatial_shape, stride=1):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.pool_spatial_shape = pool_spatial_shape
        self.stride = stride

        self.max_indices = None

    def build(self):
        self.input_shape = self.parents[0].output_shape
        # These 2 lines follow the formula in the youtube lecture
        # Giving us the output shape of this layer
        n = int((self.input_shape[0] - self.pool_spatial_shape[0]) / self.stride + 1)
        m = int((self.input_shape[1] - self.pool_spatial_shape[1]) / self.stride + 1)

        self.output_shape = (n, m, self.input_shape[-1])

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, training=False):
        windowed_images = MaxPool2D.im2window(z, self.pool_spatial_shape, self.stride)
        windowed_shape = windowed_images.shape

        flattened_spatial_dimension = (*windowed_shape[:3], windowed_shape[3] * windowed_shape[4], windowed_shape[5])

        windowed_images_with_flatten_spatial_dimension = windowed_images.reshape(flattened_spatial_dimension)
        max_indices = np.argmax(windowed_images_with_flatten_spatial_dimension, axis=-2)

        # Same as np.max(column_images, axis=-2)
        # Taken from https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        max_val = np.take_along_axis(windowed_images_with_flatten_spatial_dimension,
                                     np.expand_dims(max_indices, axis=-2), axis=-2).squeeze(axis=-2)

        if training:
            self.max_indices = max_indices

        if output_only:
            return max_val
        return max_val, max_val

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(sum, new_delta)
        raise NotImplementedError

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        return {}

    @check_built
    def update_parameters_(self, parameter_updates):
        pass

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {}
        return None

    @check_built
    def summary_(self):
        return f"MaxPooling2D", f"Output Shape {self.output_shape}"
