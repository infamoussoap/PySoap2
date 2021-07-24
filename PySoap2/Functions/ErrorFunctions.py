import numpy as np


def mse(predictions, targets, grad=False):
    if grad:
        N = len(predictions)
        return (predictions - targets) / N
    return np.mean((predictions - targets) ** 2) / 2


def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
    """ Computes cross entropy between targets (encoded as one-hot vectors) and predictions.

        Parameters
        ----------
            predictions : (N, k) np.array
            targets     : (N, k) np.array

        Returns
        -------
            float
                If grad = False then the cross_entropy score is retuned

            OR

            (N, k) np.array
                If grad = True then the gradient of the output is returned
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    if grad:
        return -targets / predictions + (1 - targets) / (1 - predictions)

    ce = -np.sum(targets * np.log(predictions + 1e-9))
    return ce


error_functions = {'mse': mse,
                   'cross_entropy': cross_entropy}
