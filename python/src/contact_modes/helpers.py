import numpy as np
import numpy.matlib
import pyhull
from scipy.optimize import linprog
import scipy as sp
from time import time
from .polytope import FaceLattice
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

def in_convex_hull(p, hull):
    num_p = p.shape[0]
    num_v = hull.shape[0]
    dim = hull.shape[1]
    res = np.zeros(num_p, dtype=bool)
    pc = np.sum(hull, axis=0) / num_v
    A = hull - pc
    p = p - pc
    for i in range(num_p):
        Ae = hull.T
        be = p[i]
        Au = np.ones((1,num_v))
        bu = 1
        bounds = tuple([(0,None) for i in range(num_v)])
        x = linprog(np.zeros(num_v), A_ub=Au, b_ub=bu, A_eq = Ae, b_eq = be, bounds = bounds)
        res[i] = x.success
    return res


def zenotope_vertex(normals):
    # N: normals of the hyperplanes
    num_normals = normals.shape[0]
    dim = normals.shape[1]
    V = np.vstack((normals[0],-normals[0]))
    Sign = np.array([[1],[-1]])
    print('num of normals: ',num_normals)
    for i in range(1,num_normals):
        print('normal',i)
        normal = normals[i]
        V_ = np.empty((0,dim))
        Sign_ = np.empty((0,i+1))

        orth = sp.linalg.orth((V - V[1, :]).T)
        if orth.shape[1] == dim:
            null = np.zeros((6,0))
        else:
            null = sp.linalg.null_space((V - V[1, :]))
        if orth.shape[1] != V.shape[1]:
            orth = sp.linalg.orth((V - V[1, :]).T)
            V_reduced = np.dot((V - V[1, :]), orth)
        else:
            orth = np.identity(V.shape[1])
            V_reduced = V - V[1, :]
        if V_reduced.shape[1] > 1:
            ret = pyhull.qconvex('n', V_reduced)
            Face = np.array([np.fromstring(ret[i], dtype=float, sep=' ') for i in range(2, len(ret))])
            A = Face[:, 0:-1]
            b = Face[:, -1]


        for k in range(V.shape[0]):
            v = V[k]
            v_pm = np.vstack((v + normal,v - normal))
            v_sign = np.hstack((np.array([Sign[k],Sign[k]]),[[1],[-1]]))
            '''
            if V_reduced.shape[1]>1:
                ind_v = np.logical_not(np.all(np.dot(A, np.dot(v_pm - V[1,:],orth).T) + np.vstack((b,b)).T <=0 ,axis=0)
                                         & np.all(np.abs(np.dot(v_pm - V[1,:],null))<1e-5,axis=1))
            else:
                ind_v = np.logical_not(in_convex_hull(v_pm, V))
            '''
            ind_v = [True,True]
            Sign_ = np.vstack((Sign_,v_sign[ind_v]))
            V_ = np.vstack((V_,v_pm[ind_v]))

        # take the convex hull of V_
        orth =  sp.linalg.orth((V_ - V_[1, :]).T)
        if orth.shape[1] != V.shape[1]:
            Vr = np.dot((V_ - V_[1, :]), orth)
        else:
            Vr = V_ - V_[1, :]
        vertices = [list(Vr[i]) for i in range(Vr.shape[0])]
        ret = pyhull.qconvex('Fx', vertices)

        ind_vertices = [int(ret[j]) for j in range(1,len(ret))]
        V = V_[ind_vertices]
        Sign = Sign_[ind_vertices]

    return V, Sign

def feasible_faces(Lattice, V, Sign, ind_feasible):
    FeasibleLattice = FaceLattice()
    Modes = []
    FeasibleLattice.L = []

    for i in range(len(Lattice.L)):
        FeasibleLattice.L.append([])
        for j in range(len(Lattice.L[i])):
            if any([k in Lattice.L[i][j].verts for k in ind_feasible]):
                sign = Sign[list(Lattice.L[i][j].verts)]
                mode_sign = np.zeros(Sign.shape[1],dtype=int)
                mode_sign[np.all(sign == 1, axis=0)] = 1
                mode_sign[np.all(sign == -1, axis=0)] = -1

                #Faces.append(face)
                Modes.append(mode_sign)
                Lattice.L[i][j].m = mode_sign
                FeasibleLattice.L[i].append(Lattice.L[i][j])
    return Modes, FeasibleLattice



