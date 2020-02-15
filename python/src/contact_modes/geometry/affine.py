import numpy as np
from numpy.linalg import norm
from scipy.linalg import orth

DEBUG=False

def project_affine_subspace(V):
    if DEBUG:
        print('project_affine_subspace')
        print('     dim(V):', V.shape)
    # Check if 0 ∈ Aff(V)
    A = np.concatenate((V, np.ones((1, V.shape[1]))), axis=0)
    b = np.zeros((A.shape[0],1))
    b[-1,0] = 1
    x = np.linalg.lstsq(A, b, None)[0]
    x0 = (A @ x)[0:-1]
    s = (A @ x)[-1]
    # Case: 0 ∈ Aff(V)
    if norm(x0) < 1e-10 and np.abs(s - 1) < 1e-10:
        V = orth(V, np.finfo(np.float32).eps).T @ V
    # Case: 0 ∉ Aff(V)
    else:
        V = V.copy()
        V -= V[:,0,None]
        V = orth(V, np.finfo(np.float32).eps).T @ V
    if DEBUG:
        print('dim(aff(V)):', V.shape)
    return V
