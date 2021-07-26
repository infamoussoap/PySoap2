import numpy as np


def accuracy(predictions, target):
    if len(target.shape) == 2 and target.shape[1] == 1:
        """ target assumed to be shape (n, 1), so argmax doesnt work """
        predicted_labels = (predictions > 0.5).astype(int)
        return np.mean(predicted_labels == target.astype(int))

    return np.mean(np.argmax(predictions, axis=-1) == np.argmax(target, axis=-1))


metric_functions = {'accuracy': accuracy}
