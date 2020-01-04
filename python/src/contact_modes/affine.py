import numpy as np


def affine_dim(x):
    return linear_dim(x - x[None,0,:])
    # lin_dim = linear_dim(x)
    # If 0 in aff(x), then dim(aff(x)) = dim(lin(x)).
    # Else if 0 not in aff(x), then dim(aff(x)) = dim(lin(x)) - 1.
    # return lin_dim - 1

def affine_basis():
    pass

def linear_dim(x):
    print(x)
    return np.linalg.matrix_rank(x)

def linear_basis():
    pass