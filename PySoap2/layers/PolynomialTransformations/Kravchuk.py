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


def polynomials(N, p):
    """ Returns the normalised Kravchuk polynomials """
    J = symmetric_jacobi(N, p)
    z, Q = eigh(J)

    Q = np.array(Q)[:, np.argsort(z)].real
    return Q * np.sign(Q[0])
