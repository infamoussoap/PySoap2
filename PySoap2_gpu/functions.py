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

    else:
        raise Exception(f'{name} is not a defined function.')
