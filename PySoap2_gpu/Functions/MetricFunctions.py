import pyopencl.array as cl_array

from PySoap2_gpu.utils import ClArrayTricks


def accuracy(predictions, target):
    N = len(target)

    if len(target.shape) == 2 and target.shape[1] == 1:
        """ target assumed to be shape (n, 1), so argmax doesnt work """
        predicted_labels = (predictions > 0.5).astype(int)
        return cl_array.sum(predicted_labels == target.astype(int)) / N

    predicted_labels = ClArrayTricks.arg_max_across_last_axis(predictions)
    target_labels = ClArrayTricks.arg_max_across_last_axis(target)
    return cl_array.sum(predicted_labels == target_labels) / N


metric_functions = {'accuracy': accuracy}
