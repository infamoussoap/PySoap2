import numpy as np

from PySoap2_gpu.utils import ClArrayTricks


def accuracy(predictions, target):
    predicted_labels = ClArrayTricks.arg_max_across_last_axis(predictions)
    target_labels = ClArrayTricks.arg_max_across_last_axis(target)

    return np.mean((predicted_labels == target_labels).get())


metric_functions = {'accuracy': accuracy}