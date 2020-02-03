# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from .exp import ExpCoefs, compute_exp_matrix3, skew3


def unit_perp3(v):
    v00 = v[0]*v[0]
    v11 = v[1]*v[1]
    v22 = v[2]*v[2]
    v01 = v[0]*v[1]
    v02 = v[0]*v[2]
    v12 = v[1]*v[2]

    p = np.zeros((3,1))
    if v00 < v11:
        if v00 < v22:
            p[0] = 1.0 - v00
            p[1] = -v01
            p[2] = -v02
        else:
            p[0] = -v02
            p[1] = -v12
            p[2] = 1.0 - v22
    elif v00 < v22:
        if v00 < v11:
            p[0] = 1.0 - v00
            p[1] = -v01
            p[2] = -v02
        else:
            p[0] = -v01
            p[1] = 1.0 - v11
            p[2] = -v12
    elif v11 < v22:
            p[0] = -v01
            p[1] = 1.0 - v11
            p[2] = -v12
    else:
        p[0] = -v02
        p[1] = -v12
        p[2] = 1.0 - v22
    
    p = p / np.linalg.norm(p)
    return p

class SO3(object):
    dim = 3
    dof = 3

    def __init__(self, R=None, dtype=np.float64):
        if R is not None:
            if len(R.shape) == 2:
                R = R.flatten()
            self.R = R
        else:
            self.R = np.zeros((9,), dtype=dtype)
        self.dtype = dtype

    def __mul__(self, other):
        return SO3.multiply(self, other)
    
    def __rmul__(self, other):
        return SO3.multiply(other, self)

    def invert(self):
        ginv = SO3.inverse(self)
        self.R = ginv.R

    def matrix(self):
        return self.R.reshape((3,3))

    def set_matrix(self, R):
        self.R = R.flatten()

    def __str__(self):
        return self.matrix().__str__()

    @staticmethod
    def ad(a):
        # algebraic adjoint
        return skew3(a).reshape((3,3))

    @staticmethod
    def identity():
        return SO3(R=np.eye(3).flatten())

    @staticmethod
    def multiply(a, b):
        # FIXME add matrix multiplication for flat arrays
        return SO3(R=np.dot(a.R.reshape((3,3)), b.R.reshape(3,3)).flatten(), dtype=a.dtype)

    @staticmethod
    def inverse(g):
        r1, r2, r5 = g.R[1], g.R[2], g.R[5]
        ginv = SO3()
        ginv.R[0] = g.R[0]
        ginv.R[1] = g.R[3]
        ginv.R[2] = g.R[6]
        ginv.R[3] = r1
        ginv.R[4] = g.R[4]
        ginv.R[5] = g.R[7]
        ginv.R[6] = r2
        ginv.R[7] = r5
        ginv.R[8] = g.R[8]
        return ginv

    @staticmethod
    def transform_point(g, x):
        return np.dot(g.R.reshape((3,3)), x)
    
    @staticmethod
    def transform_point_by_inverse(g, x):
        return np.dot(g.R.reshape((3,3)).T, x)

    @staticmethod
    def exp(x):
        theta_sq = np.dot(x.T,x)
        coefs = ExpCoefs(theta_sq, dtype=x.dtype)
        X = SO3(dtype=x.dtype)
        X.R = compute_exp_matrix3(coefs.cos_theta, coefs.A, coefs.B, x)
        return X
    
    @staticmethod
    def log(X):
        w = np.zeros((3,1), dtype=X.dtype)
        w[0] = 0.5 * (X.R[7] - X.R[5])
        w[1] = 0.5 * (X.R[2] - X.R[6])
        w[2] = 0.5 * (X.R[3] - X.R[1])

        tr = X.R[0] + X.R[4] + X.R[8]
        ct = 0.5  * (tr - 1.0)
        st2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2]

        if ct > 0.999856:
            #  Small angles
            #  Taylor expansion of f(x) = arcsin(x) / x
            #  x^2 = st2
            f = 1.0 + st2*((1.0/6.0) + st2*((3.0/40.0) + st2*(5.0/112.0)))
            w[0] *= f
            w[1] *= f
            w[2] *= f
            return w

        if ct > -0.99:
            theta = np.arccos(ct)
            st = np.sqrt(st2)
            factor = theta / st
            w[0] *= factor
            w[1] *= factor
            w[2] *= factor
            return w

        # Angles near pi
        st = np.sqrt(st2)
        theta = np.pi - np.arcsin(st)
        invB = (theta*theta) / (1.0 - ct)

        w00 = invB*(X.R[0] - ct)
        w11 = invB*(X.R[4] - ct)
        w22 = invB*(X.R[8] - ct)

        w01 = invB*0.5*(X.R[1] + X.R[3])
        w02 = invB*0.5*(X.R[2] + X.R[6])
        w12 = invB*0.5*(X.R[5] + X.R[7])

        # Take sqrt of biggest element of w
        if w00 > w11:
            if w00 > w22:
                w[0] = (-1 if w[0] < 0 else 1) * np.sqrt(w00)
                inv_w0 = 1.0/w[0]
                w[1] = w01 * inv_w0
                w[2] = w02 * inv_w0
            else:
                w[2] = (-1 if w[2] < 0 else 1) * np.sqrt(w22)
                inv_w2 = 1.0/w[2]
                w[0] = w02 * inv_w2
                w[1] = w12 * inv_w2
        elif w11 > w22:
            w[1] = (-1 if w[1] < 0 else 1) * np.sqrt(w11)
            inv_w1 = 1.0/w[1]
            w[0] = w01 * inv_w1
            w[2] = w12 * inv_w1
        else:
            w[2] = (-1 if w[2] < 0 else 1) * np.sqrt(w22)
            inv_w2 = 1.0/w[2]
            w[0] = w02 * inv_w2
            w[1] = w12 * inv_w2

        return w

    @staticmethod
    def compute_rotation_between_unit_vectors(a, b):
        cos_theta = np.dot(a, b)

        if cos_theta < -0.9:
            neg_a = -np.copy(a)
            neg_a_to_b, valid = SO3.compute_rotation_between_unit_vectors(neg_a, b)
            if not valid:
                return False

            a_to_neg_a = SO3()
            p = unit_perp3(a)
            C = 2.0
            Cp00 = C*p[0]*p[0]
            Cp11 = C*p[1]*p[1]
            Cp22 = C*p[2]*p[2]
            Cp01 = C*p[0]*p[1]
            Cp02 = C*p[0]*p[2]
            Cp12 = C*p[1]*p[2]
            a_to_neg_a.R[0] = 1.0 - (Cp11 + Cp22)
            a_to_neg_a.R[4] = 1.0 - (Cp00 + Cp22)
            a_to_neg_a.R[8] = 1.0 - (Cp00 + Cp11)
            a_to_neg_a.R[1] = Cp01
            a_to_neg_a.R[2] = Cp02
            a_to_neg_a.R[3] = Cp01
            a_to_neg_a.R[5] = Cp12
            a_to_neg_a.R[6] = Cp02
            a_to_neg_a.R[7] = Cp12

            R = SO3.multiply(neg_a_to_b, a_to_neg_a)
            return R, True

        C = 1.0 / (1.0 + cos_theta)
        
        w = np.array([a[1]*b[2] - a[2]*b[1],
                      a[2]*b[0] - a[0]*b[2],
                      a[0]*b[1] - a[1]*b[0]])

        w00 = w[0]*w[0]
        w11 = w[1]*w[1]
        w22 = w[2]*w[2]
        R = SO3()
        R.R[0] = 1.0 - C*(w11 + w22)
        R.R[4] = 1.0 - C*(w00 + w22)
        R.R[8] = 1.0 - C*(w00 + w11)
        Cw01 = C*w[0]*w[1]
        Cw02 = C*w[0]*w[2]
        Cw12 = C*w[1]*w[2]
        R.R[1] = Cw01 - w[2]
        R.R[2] = Cw02 + w[1]
        R.R[3] = Cw01 + w[2]
        R.R[5] = Cw12 - w[0]
        R.R[6] = Cw02 - w[1]
        R.R[7] = Cw12 + w[0]
        return R, True
