from sympy import *

def eye_rat(n):
    I = eye(n)
    for i in range(n):
        for j in range(n):
            I[i,j] = Rational(I[i,j])
    return I

def round2zero(m, e):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if (isinstance(m[i,j], Float) and m[i,j] < e):
                m[i,j] = 0