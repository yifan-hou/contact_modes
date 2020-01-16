import numpy as np
from quadprog import solve_qp
from scipy.linalg import null_space as null
from scipy.linalg import orth as orth
from scipy.optimize import linprog


DEBUG = False

def int_pt_cone(H):
    # Find dimension of halfspaces.
    dim = H.shape[1]

    # Add a box around the origin to ensure that the resulting linear program is
    # bounded.
    H = H.copy()
    offset = np.array([[1]])
    b = np.zeros((H.shape[0],1))
    for i in range(dim):
        for s in [1, -1]:
            n = np.zeros((dim, 1))
            n[i] = s
            H = np.concatenate((H, n.T), axis=0)
            b = np.concatenate((b, offset), axis=0)

    # Solve linear program for strictly interior point.
    n = H.shape[0]
    H = np.concatenate((H, np.ones((n,1))), axis=1)
    bounds = []
    for i in range(dim):
        bounds.append((None,None))
    bounds.append((0,None))
    c = np.zeros((dim+1,))
    c[-1] = -1.0
    x = linprog(c, H, b, bounds=bounds, method='interior-point')

    return x.x[0:dim,None]

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
