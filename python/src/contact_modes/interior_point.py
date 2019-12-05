import numpy as np
from quadprog import solve_qp
from scipy.optimize import linprog


def interior_point_halfspace(A, b):
    """Find a strictly interior point within the intersection of halfspaces.

    Arguments:
        A {np.ndarray} -- Ax - b <= 0
        b {np.ndarray} -- 
    """
    # Find dimension of halfspaces.
    dim = A.shape[1]

    # Solve QP for a point on the interior or boundary.
    G = np.eye(dim)
    a = np.zeros((dim,))
    x,_,_,_,_,_ = solve_qp(G, a, -A.T, -b.flatten())
    x = x.reshape((dim,1))

    # Add a box around the point to ensure that the resulting linear program is
    # bounded.
    A = A.copy()
    b = b.copy()
    for i in range(dim):
        for s in [1, -1]:
            n = np.zeros((dim, 1))
            n[i] = s
            d = np.dot(n.T, x) + 1
            A = np.concatenate((A, n.T), axis=0)
            b = np.concatenate((b, d), axis=0)

    # Solve linear program for strictly interior point.
    n = A.shape[0]
    A = np.concatenate((A, np.ones((n,1))), axis=1)
    bounds = []
    for i in range(dim):
        bounds.append((None,None))
    bounds.append((0,None))
    c = np.zeros((dim+1,))
    c[-1] = -1.0
    x = linprog(c, A, b, bounds=bounds)

    return x.x[0:dim,None]