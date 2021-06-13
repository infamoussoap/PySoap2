import numpy as np


def transform_1d(dataset, P1, single_image=False):
    """ Interface for the 1d and multi-1d transformation

        Parameters
        ----------
        dataset : (N, m, *) np.array
            Can either be 2d or 3d np.array
        P1 : (m, m) np.array
            The polynomial
        single_image : bool
            If true then dataset assumed to be (m, *) np.array
    """

    if single_image:
        return transform_1d(dataset[None, ...], P1, single_image=False)[0]

    if len(dataset.shape) == 2:
        return _transform_1d(dataset, P1)
    elif len(dataset.shape) == 3:
        return _multi_transform_1d(dataset, P1)
    else:
        raise ValueError(f'{dataset.shape} is not a valid shape for 1d transformation')


def _transform_1d(dataset, P1):
    """ Dataset assumed to be (N, m) dimension np.array """
    return dataset @ P1.T[None, :]


def _multi_transform_1d(dataset, P1):
    """ Dataset assumed to be (N, m, k) dimension np.array, with k being the 'color' dimension """
    k = dataset.shape[2]

    out = np.zeros_like(dataset)
    for i in range(k):
        out[:, :, i] = dataset[:, :, i] @ P1.T[None, :]
    return out


def transform_2d(dataset, P1, P2=None, single_image=False):
    """ This is the interface for the 2d and multi-2d transformations

        Parameters
        ----------
        dataset : (N, i, j, *) np.array
            Can be either 3d or 4d np.array
        P1 : (i, i) np.array
        P2 : (j, j) np.array, optional
            If P2 is None, then P2 is assumed to be the same as P1
        single_image : bool
            If True, then dataset assumed to be (i, j, *) np.array

        Raises
        ------
        ValueError
            If dataset is not 3d or 4d np.array
    """

    if single_image:
        return transform_2d(dataset[None, ...], P1=P1, P2=P2, single_image=False)[0]

    if P2 is None:
        P2 = P1.copy()

    if len(dataset.shape) == 3:
        return _transform_2d(dataset, P1, P2)
    elif len(dataset.shape) == 4:
        return _transform_2d_multi(dataset, P1, P2)
    else:
        raise ValueError(f'{dataset.shape} is not a valid shape for 2d transformation')


def _transform_2d(dataset, P1, P2):
    """ Dataset assumed to be (N, i, j) dimension np.array """
    return P1[None, :, :] @ dataset @ P2.T[None, :, :]


def _transform_2d_multi(dataset, P1, P2):
    """ Dataset assumed to be (N, m, n, k) dimension np.array """
    k = dataset.shape[3]

    transformation = np.zeros(dataset.shape)
    for i in range(k):
        transformation[:, :, :, i] = P1[None, :, :] @ dataset[:, :, :, i] @ P2.T[None, :, :]
    return transformation
