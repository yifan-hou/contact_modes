import numpy as np
from numpy.linalg import norm

from scipy.linalg import null_space as null
from scipy.linalg import orth


DEBUG = True

def affine_dim(x):
    return linear_dim(x - x[None,0,:])
    # lin_dim = linear_dim(x)
    # If 0 in aff(x), then dim(aff(x)) = dim(lin(x)).
    # Else if 0 not in aff(x), then dim(aff(x)) = dim(lin(x)) - 1.
    # return lin_dim - 1

def affine_basis():
    pass

def proj_affine(pts):
    if DEBUG:
        print('pts')
        print(pts)
    # Check if 0 ∈ Aff(pts)
    A = np.concatenate((pts, np.ones((1,pts.shape[1]))), axis=0)
    b = np.zeros((A.shape[0],1))
    b[-1,0] = 1
    x = np.linalg.lstsq(A, b, None)[0]
    x0 = (A @ x)[0:-1]
    s = (A @ x)[-1]
    # Case: 0 ∈ Aff(pts)
    if DEBUG:
        print('norm(x0) < 1e-10')
        print(norm(x0), '<', 1e-10)
        print('np.abs(s - 1) < 1e-10')
        print(np.abs(s - 1), '<', 1e-10)
    if norm(x0) < 1e-10 and np.abs(s - 1) < 1e-10:
        if DEBUG:
            print('0 in Aff(pts)')
            print('rank')
            print(np.linalg.matrix_rank(pts))
            print('orth(pts)')
            print(orth(pts, np.finfo(np.float32).eps))
        pts = orth(pts, np.finfo(np.float32).eps).T @ pts
        if DEBUG:
            print('proj rank')
            print(np.linalg.matrix_rank(pts, np.finfo(np.float32).eps))
    # Case: 0 ∉ Aff(pts)
    else:
        pts = pts.copy()
        pts -= pts[:,0,None]
        if DEBUG:
            print('0 not in Aff(pts)')
            print('pts')
            print(pts)
            print('rank')
            print(np.linalg.matrix_rank(pts))
            print('orth(pts)')
            print(orth(pts, np.finfo(np.float32).eps))
        pts = orth(pts, np.finfo(np.float32).eps).T @ pts
        if DEBUG:
            print('proj rank')
            print(np.linalg.matrix_rank(pts, np.finfo(np.float32).eps))
    if DEBUG:
        print('proj pts')
        print(pts)
    return pts
    
def linear_dim(x):
    print(x)
    return np.linalg.matrix_rank(x)

def linear_basis():
    pass

def affine_dep(X):
    n_cols = X.shape[1]
    return null(np.concatenate((X, np.ones((1, n_cols))), axis=0))