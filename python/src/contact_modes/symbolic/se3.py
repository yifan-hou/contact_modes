import sympy as sp

from sympy import sqrt, nsimplify, pprint, Rational
from sympy.matrices import Matrix, eye, zeros, ones

from .so3 import *

DEBUG = False

def hat6(xi):
    xi_hat = zeros(4, 4)
    v = xi[0:3,0]
    w = xi[3:6,0]
    xi_hat[0:3,0:3] = hat3(w)
    xi_hat[0:3,3] = v
    return xi_hat

def exp6(xi, theta):
    # xi = xi / xi.norm()
    v = xi[0:3,0]
    w = xi[3:6,0]
    R = exp3(w, theta)
    g = zeros(4, 4)
    g[0:3,0:3] = R
    g[0:3,3] = (eye(3) - R)*(w.cross(v)) + w*w.dot(v)*theta
    g[3,3] = 1
    return g