from __future__ import division
import numpy as np
import math


def genvec_SO3(theta=np.pi):
    x = theta*(np.random.rand(3,1)*2.0 - 1.0)
    xx = np.dot(x.T, x)
    f = 0.999999
    max_theta = f * np.pi
    if xx >= max_theta * max_theta:
        f = max_theta / np.sqrt(xx)
        x *= f
    return x

def genvec_SE3():
    x = np.random.rand(6,1)*3.0-1.5
    x[3:6] = genvec_SO3()
    return x

def skew3(a):
    skew_a = np.zeros((9,), dtype=np.float64)
    skew_a[0], skew_a[4], skew_a[8] = 0, 0, 0
    skew_a[1] = -a[2]
    skew_a[2] = a[1]
    skew_a[3] = a[2]
    skew_a[5] = -a[0]
    skew_a[6] = -a[1]
    skew_a[7] = a[0]
    return skew_a

def epsilon(dtype=np.float64):
    if dtype == np.float64:
        return 1.11e-16
    elif dtype == np.float32:
        return 5.96e-8

def sqrt_epsilon(dtype=np.float64):
    if dtype == np.float64:
        return 1.054e-8
    elif dtype == np.float32:
        return 2.44e-4

def gamma(n):
    return (n * epsilon()) / (1 - n * epsilon())

def compute_exp_matrix3(a, b, c, w):
    # Compute m = A*I + B*wx + C*ww'
    m0 = np.zeros((3,), dtype=np.float64)
    m1 = np.zeros((3,), dtype=np.float64)
    m2 = np.zeros((3,), dtype=np.float64)

    Cw0 = c * w[0]
    Cw1 = c * w[1]
    Cw2 = c * w[2]
    m0[0] = a + Cw0 * w[0]
    m1[1] = a + Cw1 * w[1]
    m2[2] = a + Cw2 * w[2]

    Cw01 = Cw0 * w[1]
    Cw02 = Cw0 * w[2]
    Cw12 = Cw1 * w[2]
    Bw0 = b * w[0]
    Bw1 = b * w[1]
    Bw2 = b * w[2]

    m0[1] = Cw01 - Bw2
    m0[2] = Cw02 + Bw1
    m1[0] = Cw01 + Bw2
    m1[2] = Cw12 - Bw0
    m2[0] = Cw02 - Bw1
    m2[1] = Cw12 + Bw0

    m = np.vstack((m0, m1, m2))
    return m.flatten()

def compute_exp_coefs_small(theta_sq, coefs):
    tt = theta_sq
    coefs.B = 0.5 - tt*((1.0/24.0) - tt*(1.0/720.0))
    coefs.cos_theta = 1.0 - tt*coefs.B
    coefs.C = (1.0/6.0) - tt*((1.0/120.0) - tt*(1.0/5040.0))
    coefs.A = 1.0 - tt*coefs.C

def compute_exp_coefs_large(theta_sq, coefs):
    theta = np.sqrt(theta_sq)
    inv_tt = 1.0 / theta_sq
    coefs.A = np.sin(theta)/theta
    coefs.cos_theta = np.cos(theta)
    coefs.B = (1.0 - coefs.cos_theta) * inv_tt
    coefs.C = (1.0 - coefs.A) * inv_tt
    return inv_tt

class ExpCoefs(object):
    def __init__(self, theta_sq=None, dtype=np.float64):
        self.A = None 
        self.B = None
        self.C = None
        self.cos_theta = None
        self.dtype = dtype
        if theta_sq is not None:
            self.compute(theta_sq)

    def compute(self, theta_sq):
        if theta_sq < 25 * sqrt_epsilon(self.dtype):
            compute_exp_coefs_small(theta_sq, self)
        else:
            compute_exp_coefs_large(theta_sq, self)