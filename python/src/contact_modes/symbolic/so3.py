import sympy as sp

from sympy.matrices import Matrix, eye, zeros, ones
from sympy import Rational, sqrt, pprint


def hat3(w):
    w_hat = zeros(3,3)
    w_hat[0,1] = -w[2]
    w_hat[0,2] =  w[1]
    w_hat[1,0] =  w[2]
    w_hat[1,2] = -w[0]
    w_hat[2,0] = -w[1]
    w_hat[2,1] =  w[0]
    return w_hat

def exp3(w, theta):
    # w = w / w.norm()
    w_hat = hat3(w)
    R = eye(3) + sp.sin(theta) * w_hat + (1 - sp.cos(theta)) * w_hat * w_hat
    return R