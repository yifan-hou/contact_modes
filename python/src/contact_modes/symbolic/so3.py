import sympy as sym

from sympy.matrices import Matrix, eye, zeros, ones


def hat3(w):
    w_hat = zeros(3,3)
    w_hat[0,1] = -w[2]

def exp(w):
    pass