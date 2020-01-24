# -*- coding: utf-8 -*-
import numpy as np
from .so3 import SO3
from .exp import ExpCoefs, compute_exp_matrix3, epsilon, skew3


class SE3(object):
    dim = 3
    dof = 6

    def __init__(self, R=None, t=None, dtype=np.float64):
        self.R = R
        if self.R is None:
            self.R = SO3()
        self.t = t
        if self.t is None:
            self.t = np.zeros((3,1))
        self.dtype = dtype

    def __mul__(self, other):
        return SE3.multiply(self, other)
    
    def __rmul__(self, other):
        return SE3.multiply(other, self)

    def invert(self):
        ginv = SE3.inverse(self)
        self.R = ginv.R
        self.t = ginv.t

    def matrix(self):
        tf = np.eye(4)
        tf[0:3,0:3] = self.R.R.reshape((3,3))
        tf[0:3,3,None] = self.t
        return tf

    def set_matrix(self, tf):
        self.R.set_matrix(tf[0:3,0:3])
        self.t = tf[0:3,3,None]

    def set_translation(self, t):
        self.t = t.reshape((3,1))
    
    def set_rotation(self, R):
        self.R.set_matrix(R)

    def __str__(self):
        return self.matrix().__str__()

    # def __repr__(self):
    #     return self.__str__()

    @staticmethod 
    def ad(a):
        # exp(ad[a]) → Ad[exp(a)]
        ada = np.zeros((6,6))
        ada[0:3,0:3] = skew3(a[3:6]).reshape((3,3))
        ada[0:3,3:6] = skew3(a[0:3]).reshape((3,3))
        ada[3:6,3:6] = skew3(a[3:6]).reshape((3,3))
        return ada.flatten()

    @staticmethod
    def Ad(g):
        # Ad[g] → [R ̂tR; 0 R]
        adjoint = np.zeros((6,6))
        adjoint[0:3,0:3] = g.R.R.reshape((3,3))
        adjoint[0:3,3:6] = np.dot(SO3.ad(g.t).reshape((3,3)), g.R.R.reshape((3,3)))
        adjoint[3:6,3:6] = g.R.R.reshape((3,3))
        return adjoint

    @staticmethod
    def identity():
        return SE3(R=SO3.identity())

    @staticmethod
    def multiply(a, b):
        # FIXME add matrix multiplication for vectors
        ab = SE3()
        ab.t = a.t + np.dot(a.R.matrix(), b.t)
        ab.R = SO3.multiply(a.R, b.R)
        return ab

    @staticmethod
    def inverse(g):
        ginv = SE3()
        ginv.R = SO3.inverse(g.R)
        ginv.t = -SO3.transform_point(ginv.R, g.t)
        return ginv

    @staticmethod
    def transform_point(g, x):
        return g.t + SO3.transform_point(g.R, x)

    @staticmethod
    def transform_point_by_inverse(g, x):
        y = x - g.t
        return SO3.transform_point_by_inverse(g.R, y)

    @staticmethod
    def velocity_at_point(x, pt):
        u = x[0:3].reshape((3,1))
        w = x[3:6].reshape((3,1))
        return np.cross(w, pt, axis=0) + u

    @staticmethod
    def exp(x):
        x = x.reshape((6,1))

        X = SE3()

        u = x[0:3]
        w = x[3:6]
        coefs = ExpCoefs(np.dot(w.T, w))
        X.R.R = compute_exp_matrix3(coefs.cos_theta, coefs.A, coefs.B, w)

        wxu = np.cross(w, u, axis=0)
        wxwxu = np.cross(w, wxu, axis=0)
        X.t = u + coefs.B*wxu + coefs.C*wxwxu

        return X

    @staticmethod
    def log(X):
        w = SO3.log(X.R)
        theta_sq = np.dot(w.T, w)
        coefs = ExpCoefs(theta_sq)
        a = coefs.A
        b = coefs.B
        c = coefs.C

        d = 0
        if theta_sq < epsilon(X.dtype)*25:
            d = 1.0/12.0 + theta_sq*(1.0/720.0 + theta_sq*1.0/30240.0)
        elif theta_sq > 9.0:
            d = (b - 0.5*a) / (b*theta_sq)
        else:
            d = (b*0.5 - c) / a
        
        wxt = np.cross(w, X.t, axis=0)
        wxwxt = np.cross(w, wxt, axis=0)
        u = X.t - 0.5*wxt + d*wxwxt

        uw = np.array([u, w]).reshape((6,1))
        return uw
