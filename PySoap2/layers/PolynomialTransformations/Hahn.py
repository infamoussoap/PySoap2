import numpy as np
from scipy.sparse import dia_matrix
from scipy.special import gammaln
from numpy.linalg import eigh


def Id(n):
    return np.identity(n + 1)


def x(n, i=0):
    return np.arange(i, n + 1)


def E(n, a, b, i=0):
    return x(n, i=i) * (x(n, i=i) + a + b + 1)


def grid(n, a, b, dual=False):
    if dual:
        return E(n, a, b)
    return x(n)


def weight(n, a, b, dual=False):
    """Computes the weights for Hahn or Dual Hahn polynomials.
        The weights are normalised so that sum(weight) = 1.

        Parameters
        ----------
        n   (int)        : max degree of either set of polynomials
        a,b (float > -1) : polynomial type parameters
        dual      (T/F)  : Standard Hahn, or Dual Hahn

        """

    ab, nb, na, nab = a + b, n + b, n + a, n + a + b

    def l(n):
        return gammaln(n + 1)

    if not dual:
        W = l(ab + 1) + l(n) + l(x(n) + a) + l(nb - x(n)) \
            - (l(a) + l(b) + l(x(n)) + l(n - x(n)) + l(nab + 1))

    if dual:
        W = l(a + x(n)) + l(1 + ab + x(n)) + l(n) + l(nb) \
            - (l(x(n)) + l(n - x(n)) + l(b + x(n)) + l(nab + x(n) + 1) + l(a))

        # avoids problems when a+b = -1
        W[1:] += np.log((2 * x(n, i=1) + ab + 1) / (x(n, i=1) + ab + 1))

    return np.exp(W)


def Jacobi(n, a, b, symmetric=True, dual=False):
    """Computes the Jacobi operator for Hahn or Dual Hahn polynomials

    Parameters
    ----------
    n   (int)        : max degree of either set of polynomials
    a,b (float > -1) : polynomial type parameters
    symmetric (T/F)  : symmeterise output
    dual      (T/F)  : Standard Hahn, or Dual Hahn

    """

    ab, na, nb, nab, nnab = a + b, n + a, n + b, n + a + b, 2 * n + a + b

    def k(i, j, s=1):
        return np.arange(i, j + np.sign(s), s)

    if not dual:
        low = k(1, n) * k(b + 1, nb) * k(nab + 2, nnab + 1) / (k(ab + 2, nnab, 2) * k(ab + 3, nnab + 1, 2))
        upp = k(n, 1, -1) * k(a + 1, na) / k(ab + 2, nnab + 1, 2)
        upp[1:] *= k(ab + 2, nab) / k(ab + 3, nnab, 2)  # avoids problems when a+b = -1

    if dual:
        low = k(1, n) * k(nb, b + 1, -1)
        upp = k(n, 1, -1) * k(a + 1, na)

    mid = np.zeros(n + 1)
    mid[:-1] = upp
    mid[1:] += low

    if symmetric: low = upp = np.sqrt(low * upp)

    low, upp = np.insert(low, n, 0), np.insert(upp, 0, 0)

    return dia_matrix(([-low, mid, -upp], [-1, 0, 1]), shape=(n + 1, n + 1))


def polynomials(n, a, b, symmetric=True, dual=False, norm=True, eps=1e-6):
    """Returns an orthogonal matrix of scaled Hanh/dual-Hanh polynomials.
        Q[k,x] gives the kth degree at the xth grid point.

            Q[k,x] = sqrt(weight[k]*weight[x]/weight[0])*P[k,x]

            Q.T @ Q = Identity (grid space)
            Q @ Q.T = Identity (degree space)

        where P[k,x] is a polynomial in both k, and x, with
        P[0,x] = P[k,x] = 1.

        We already know the eigenvalues, z = grid(n,a,b,dual=dual)

        Warnings:

            1) Q[dual=False/True].T = Q[dual=True/False] @ S,
               where S is a diagonal martix of +-1.
               Fixing this is harder for large n, a, b.

            2) dual=True is less accurate than dual=False for large n.

            3) Both options can lose accuracy for large enough n.

        Parameters
        ----------
        n   (int)        : max degree of either set of polynomials
        a,b (float > -1) : polynomial type parameters
        symmetric (T/F)  : symmeterise output
        dual      (T/F)  : Standard Hahn, or Dual Hahn

        """
    if not norm:
        return unnormalised(n, a, b, symmetric=symmetric, dual=dual, eps=eps)

    z, Q = eigh(Jacobi(n, a, b, symmetric=symmetric, dual=dual).todense())

    Q = np.array(Q)[:, np.argsort(z)].real
    Q *= np.sign(Q[0])[None, :]

    # The rows are Q[k,x] can still sometimes
    # be multiplied by random +-1.
    # This tries to fix the sign problem.
    # I'm not 100% sure it always works.
    # But it seems to.
    # This step can be slow.
    # It can be optimised.

    K = Jacobi(n, a, b, symmetric=symmetric, dual=(not dual))
    e = 1 / grid(n, a, b, dual=(not dual))[1:]

    i, j = [np.inf], n
    while (len(i) > 0) and (j > -1):

        i = np.where(np.abs(((Q[1:] @ K).T * e).T - Q[1:]) > eps)[1]
        i = np.sort(np.array(list(set(i)))) + 1

        if len(i) > 0: Q[:, i[0]] *= -1

        j -= 1

    return Q


def unnormalised(n, a, b, symmetric=True, dual=False, eps=1e-6):
    Q = polynomials(n, a, b, symmetric, dual, eps)

    w1 = weight(n, a, b)
    Q1 = np.zeros((n + 1, n + 1))

    for k in range(n + 1):
        for x1 in range(n + 1):
            Q1[k, x1] = Q[k, x1] * np.sqrt(w1[0] / (w1[k] * w1[x1]))
    for k in range(n + 1):
        Q1[k] /= Q1[k, 0]

    return Q1
