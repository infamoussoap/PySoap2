import numpy as np
import pyopencl.array as cl_array

from PySoap2_gpu.utils import ClArrayTricks


def accuracy(predictions, target):
    """ Accuracy of the predictions

        Parameters
        ----------
        predictions : (n, ...) cl_array.Array
        target : (n, ...) cl_array.Array

        Returns
        -------
        cl_array.Array
    """
    N = len(target)

    if len(target.shape) == 2 and target.shape[1] == 1:
        """ target assumed to be shape (n, 1), so argmax doesnt work """
        predicted_labels = (predictions > 0.5).astype(np.int8)
        correct_labels = predicted_labels == target.astype(np.int8)

        return cl_array.sum(correct_labels.astype(np.int32)) / N

    predicted_labels = ClArrayTricks.arg_max_across_last_axis(predictions)
    target_labels = ClArrayTricks.arg_max_across_last_axis(target)
    correct_labels = predicted_labels == target_labels
    return cl_array.sum(correct_labels.astype(np.int32)) / N


metric_functions = {'accuracy': accuracy}
