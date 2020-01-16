import numpy as np
import numpy.matlib
import pyhull
from scipy.optimize import linprog
import scipy as sp

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

class LexographicCombinations:
    def __init__(self, n, t):
        self.n = n
        self.t = t
        assert(0 <= t and t <= n)

    def __iter__(self):
        # Initialize combination vector.
        self.c = [0] * (self.t + 2)
        for j in range(self.t):
            self.c[j] = j
        self.c[self.t] = self.n
        self.c[self.t+1] = 0
        self.done = False

        return self

    def __next__(self):
        # L4 [Done?]
        if self.done:
            raise StopIteration
        # L2 [Visit]
        c = self.c.copy()
        # L3 [Find j]
        j = 1
        while self.c[j-1] + 1 == self.c[j]:
            self.c[j-1] = j - 1
            j = j + 1
        # L4 [Done?]
        if j > self.t:
            self.done = True
        # L5 [Increase c_j]
        self.c[j-1] = self.c[j-1] + 1
        # L2 [Visit]
        return c[0:self.t]

def lexographic_combinations(n, t):
    if t > n:
        return []
    return LexographicCombinations(n, t)

def exp_comb(m,n):
    # m: number of contacts
    # n: number of modes
    c = np.zeros((m,n**m),dtype=int)
    for i in range(m):
        c_i = np.array([k*np.ones((n**(m-i-1))) for k in range(n)])
        c[i] = np.matlib.repmat(c_i.flatten(),1,n**i)
    return c.T

def in_hull(p, hull):
    dim = hull.shape[1]
    num_p = p.shape[0]
    num_v = hull.shape[0]

    pc = np.sum(hull, axis=0) / num_v
    A = hull - pc
    p = p - pc

    M = np.concatenate((p,A),axis=0)
    rank_M = np.linalg.matrix_rank(M)
    rank_hull = np.linalg.matrix_rank(A)

    if rank_M < dim:
        B = sp.linalg.orth(M.T)
        #Q = np.dot(np.dot(B,np.linalg.inv(np.dot(B.T,B))),B.T)
        #M_ = np.dot(M,Q)
        M_ = np.linalg.pinv(B).dot(M.T).T
        p = M_[0:num_p,0:rank_M]
        A = M_[num_p:,0:rank_M]

    res = np.zeros(num_p,dtype=bool)
    for i in range(num_p):
        mask = np.hstack((np.zeros(num_p,dtype=bool),np.ones(num_v,dtype=bool)))
        mask[i] = True
        if rank_hull < np.linalg.matrix_rank(M[mask]):
            res[i] = False
        else:
            x = linprog(-p[i],A_ub=A, b_ub=np.ones(num_v))
            res[i] = -x.fun<1

    return res

def zenotope_vertex(normals):
    # N: normals of the hyperplanes
    num_normals = normals.shape[0]
    dim = normals.shape[1]
    V = np.vstack((normals[0],-normals[0]))
    Sign = np.array([[1],[-1]])
    for i in range(1,num_normals):
        normal = normals[i]
        V_ = np.empty((0,dim))
        Sign_ = np.empty((0,i+1))
        for k in range(V.shape[0]):
            v = V[k]
            v_pm = np.vstack((v + normal,v - normal))
            v_sign = np.hstack((np.array([Sign[k],Sign[k]]),[[1],[-1]]))
            ind_v = np.logical_not(in_hull(v_pm, V))
            Sign_ = np.vstack((Sign_,v_sign[ind_v]))
            V_ = np.vstack((V_,v_pm[ind_v]))
            
        V = V_
        Sign = Sign_

    return V, Sign
