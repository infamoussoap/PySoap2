import pyopencl as cl
import pyopencl.array as cl_array


def get_activation_function(name, gpu_context):
    if gpu_context is None:
        raise ValueError('Context for gpu cannot be None.')

    if name == 'relu':
        elementwise_relu = cl.elementwise.ElementwiseKernel(gpu_context, "float *x, float *out",
                                                            "out[i] = x[i] > 0 ? x[i] : 0.0", "relu")
        elementwise_relu_grad = cl.elementwise.ElementwiseKernel(gpu_context, "float *x, float *out",
                                                                 "out[i] = x[i] > 0 ? 1.0 : 0.0", "relu")

        def relu(x_device, grad=False):
            """ x is assumed to be an instance of cl.array.Array"""
            out_device = cl_array.empty_like(x_device)
            if grad:
                elementwise_relu_grad(x_device, out_device)
            else:
                elementwise_relu(x_device, out_device)
            return out_device

        return relu

    elif name == 'sigmoid':
        elementwise_sigmoid = cl.elementwise.ElementwiseKernel(gpu_context,
                                                               "float *x, float *out",
                                                               """
                                                               out[i] = SIGMOID(x[i])
                                                               """,
                                                               "sigmoid",
                                                               preamble='#define SIGMOID(x) x > 0 ? '
                                                                        '1.0/(1.0 + exp(-x)) : exp(x) / (exp(x) + 1.0))'
                                                               )

        def sigmoid(x_device, grad=False):
            out_device = cl_array.empty_like(x_device)
            elementwise_sigmoid(x_device, out_device)
            if grad:
                return out_device * (1 - out_device)
            return out_device

        return sigmoid

    else:
        raise Exception(f'{name} is not a defined function.')
