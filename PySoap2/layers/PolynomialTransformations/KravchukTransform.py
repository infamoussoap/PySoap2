import numpy as np

import scipy.sparse as sparse
from numpy.linalg import eigh


def symmetric_jacobi(N, p):
    """ Returns the symmetric Jacobi Matrix for the Kravchuk Polynomials """
    q = 1 - p

    upper = p * (N - np.arange(-1, N))
    lower = q * np.arange(1, N + 2)

    mid = q * np.arange(N + 1) + p * (N - np.arange(N + 1))

    upper[1:] = lower[:-1] = np.sqrt(upper[1:] * lower[:-1])  # Make matrix symmetric

    return sparse.dia_matrix(([-upper, mid, -lower], [1, 0, -1]), shape=(N + 1, N + 1)).todense()


def kravchuk_polynomials(N, p):
    """ Returns the normalised Kravchuk polynomials """
    J = symmetric_jacobi(N, p)
    z, Q = eigh(J)

    Q = np.array(Q)[:, np.argsort(z)].real
    return Q * np.sign(Q[0])


def transform_1d(dataset, p=0.5, single_image=False, inverse=False):
    """ Interface for the 1d and multi-1d transformation

        dataset assumed to be (N, ...) np.array
    """

    if single_image:
        return transform_1d(dataset[None, ...], p=p, single_image=False, inverse=inverse)[0]

    if len(dataset.shape) == 2:
        return _transform_1d(dataset, p=p, inverse=inverse)
    elif len(dataset.shape) == 3:
        return _multi_transform_1d(dataset, p=p, inverse=inverse)
    else:
        raise ValueError(f'{dataset.shape} is not a valid shape for 1d transformation')


def _transform_1d(dataset, p=0.5, inverse=False):
    """ Dataset assumed to be (N, m) dimension np.array """
    m = dataset.shape[1]

    P = kravchuk_polynomials(m - 1, p=p)

    if inverse:
        P = P.T

    return dataset @ P.T[None, :]


def _multi_transform_1d(dataset, p=0.5, inverse=False):
    """ Dataset assumed to be (N, m, k) dimension np.array, with k being the 'color' dimension """
    m = dataset.shape[1]
    k = dataset.shape[2]

    P = kravchuk_polynomials(m - 1, p=p)

    if inverse:
        P = P.T

    out = np.zeros_like(dataset)
    for i in range(k):
        out[:, :, i] = dataset[:, :, i] @ P.T[None, :]
    return out


def transform_2d(dataset, p=0.5, single_image=False, inverse=False):
    """ This is the interface for the 2d and multi-2d transformations """

    if single_image:
        return transform_2d(dataset[None, ...], p=p, single_image=False, inverse=inverse)[0]

    if len(dataset.shape) == 3:
        return _transform_2d(dataset, p=p, inverse=inverse)
    elif len(dataset.shape) == 4:
        return _transform_2d_multi(dataset, p=p, inverse=inverse)
    else:
        raise ValueError(f'{dataset.shape} is not a valid shape for 2d transformation')


def _transform_2d(dataset, p=0.5, inverse=False):
    """ Dataset assumed to be (N, m, n) dimension np.array """
    m = dataset.shape[1]
    n = dataset.shape[2]

    P1 = kravchuk_polynomials(m - 1, p=p)
    P2 = P1 if m == n else kravchuk_polynomials(n - 1, p=p)

    if inverse:
        P1 = P1.T
        P2 = P2.T

    return P1[None, :, :] @ dataset @ P2.T[None, :, :]


def _transform_2d_multi(dataset, p=0.5, inverse=False):
    """ Dataset assumed to be (N, m, n, k) dimension np.array """
    m, n, k = dataset.shape[1], dataset.shape[2], dataset.shape[3]

    P1 = kravchuk_polynomials(m - 1, p=p)
    P2 = P1 if m == n else kravchuk_polynomials(n - 1, p=p)

    if inverse:
        P1 = P1.T
        P2 = P2.T

    transformation = np.zeros(dataset.shape)
    for i in range(k):
        transformation[:, :, :, i] = P1[None, :, :] @ dataset[:, :, :, i] @ P2.T[None, :, :]
    return transformation


def transform_3d(dataset, p=0.5, inverse=False):
    """ Dataset assumed to be (N, m, n, k) dimension np.array """
    m, n, k = dataset.shape[1], dataset.shape[2], dataset.shape[3]

    P1 = kravchuk_polynomials(m - 1, p=p)
    P2 = P1 if m == n else kravchuk_polynomials(n - 1, p=p)
    P3 = kravchuk_polynomials(k - 1, p=p)

    if inverse:
        P1 = P1.T
        P2 = P2.T
        P3 = P3.T

    return np.einsum('ia,jb,kc,nijk->nabc', P1, P2, P3, dataset, optimize='greedy')
