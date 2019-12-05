import numpy as np


def hat_2d():
    pass

def hat_3d():
    pass

def hat(w):
    W = np.zeros((3,3))
    W[0,1] = -w[2]
    W[0,2] =  w[1]
    W[1,0] =  w[2]
    W[1,2] = -w[0]
    W[2,0] = -w[1]
    W[2,1] =  w[0]
    return W

def halfspace_inequality(points, normals):
    pass