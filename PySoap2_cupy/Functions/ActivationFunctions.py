import cupy as cp


def sigmoid(x, grad=False):
    """ Stable sigmoid function for positive and negative values of x

        Parameters
        ----------
        x : np.array
        grad : bool
            If true, returns the gradient of the sigmoid

        Returns
        -------
        np.array
            Same shape as x, the input
    """
    positive = x >= 0
    negative = ~positive

    result = cp.zeros_like(x, dtype=cp.float64)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    if grad:
        return result * (1 - result)
    return result


def _positive_sigmoid(x):
    """ Stable sigmoid function for positive values """
    return 1 / (1 + cp.exp(-x))


def _negative_sigmoid(x):
    """ Stable sigmoid function for negative values """
    exp = cp.exp(x)
    return exp / (exp + 1)


def relu(x, grad=False):
    if grad:
        return (x > 0).astype(int)

    x = cp.array(x)
    x[x < 0] = 0
    return x


def softmax(x, grad=False):
    if grad:
        softmax_val = softmax(x, grad=False)
        return softmax_val * (1 - softmax_val)

    z = x - cp.max(x, axis=-1, keepdims=True)
    numerator = cp.exp(z)
    denominator = cp.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator


def tanh(x, grad=False):
    result = cp.tanh(x)
    if grad:
        return 1 - result ** 2
    return result


def linear(x, grad=False):
    if grad:
        return 1
    return x


activation_functions = {'relu': relu,
                        'softmax': softmax,
                        'linear': linear,
                        'sigmoid': sigmoid,
                        'tanh': tanh}
