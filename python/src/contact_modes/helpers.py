import numpy as np
import numpy.matlib
import pyhull
from scipy.optimize import linprog
import scipy as sp
from time import time
from .polytope import FaceLattice
from scipy.spatial import ConvexHull
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

    for i in range(1,num_normals):

        normal = normals[i]

        V_ = np.vstack((V+normal, V-normal))
        Sign_ = np.ones((2*Sign.shape[0],Sign.shape[1]+1),dtype=int)
        Sign_[:,0:Sign.shape[1]] = np.vstack((Sign,Sign))
        Sign_[Sign.shape[0]:,-1] = -1

        # take the convex hull of V_
        orth =  sp.linalg.orth((V_ - V_[1, :]).T)
        if orth.shape[1] != V.shape[1]:
            Vr = np.dot((V_ - V_[1, :]), orth)
        else:
            Vr = V_ - V_[1, :]
        vertices = [list(Vr[i]) for i in range(Vr.shape[0])]

        ret = pyhull.qconvex('Fv', vertices)
        #ind_vertices = [int(ret[j]) for j in range(1, len(ret))]
        ind_vertices = []
        for i in range(1, len(ret)):
            ind_vertices = ind_vertices + [int(x) for x in ret[i].split(' ')][1:]
        ind_vertices = np.unique(ind_vertices)

        V = V_[ind_vertices]
        Sign = Sign_[ind_vertices]

    #
    # facets = np.zeros((int(ret[0]),len([int(x) for x in ret[1].split(' ')][1:])),dtype=int)
    # for i in range(1, len(ret)):
    #     facets[i-1] = [int(x) for x in ret[i].split(' ')][1:]
    # ret_facets = np.array(facets)
    #
    # for i in range(len(ind_vertices)):
    #     ret_facets[ret_facets == ind_vertices[i]] = i

    return V, Sign#, ret_facets

def zenotope_add(V, Sign, normals):
    # N: normals of the hyperplanes
    num_normals = normals.shape[0]
    dim = normals.shape[1]

    for i in range(num_normals):

        normal = normals[i]
        V_ = np.empty((0,dim))
        Sign_ = np.empty((0,i+Sign.shape[1]))

        for k in range(V.shape[0]):
            v = V[k]
            v_pm = np.vstack((v + normal,v - normal))
            v_sign = np.hstack((np.array([Sign[k],Sign[k]]),[[1],[-1]]))
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

def get_lattice_mode(Lattice, Sign):

    Modes = []
    for i in range(len(Lattice.L)):
        for j in range(len(Lattice.L[i])):
            if len(Lattice.L[i][j].verts) != 0:
                sign = Sign[list(Lattice.L[i][j].verts)]
                mode_sign = np.zeros(Sign.shape[1],dtype=int)
                mode_sign[np.all(sign == 1, axis=0)] = 1
                mode_sign[np.all(sign == -1, axis=0)] = -1

                #Faces.append(face)
                Modes.append(mode_sign)
                Lattice.L[i][j].m = mode_sign

    return Modes

def vertex2lattice(V):

    orth = sp.linalg.orth((V - np.mean(V, 0)).T)
    dim_V = orth.shape[1]
    n_vert = V.shape[0]
    if n_vert == 2 or n_vert == 1:
        M = np.ones((n_vert,1),int)
        L = FaceLattice(M, dim_V)
        return L

    # project V into affine space
    if dim_V != V.shape[1]:
        V = np.dot((V - np.mean(V,0)), orth)

    vertices = [list(V[i]) for i in range(n_vert)]
    ret = pyhull.qconvex('s Fv', vertices)
    #print(np.array(ret))

    # get the convex hull of V
    # select faces with desired modes
    # Build facet-vertex incidence matrix.

    n_facets = int(ret[0])
    M = np.zeros((n_vert , n_facets), int)
    for i in range(1, len(ret)):
        vert_set = [int(x) for x in ret[i].split(' ')][1:]
        for v in vert_set:
            M[v,i-1] = 1

    # Build face lattice.
    L = FaceLattice(M, dim_V)
    return L

def to_lattice(V,facets):
    n_vert = V.shape[0]
    n_facets = facets.shape[0]
    dim_V = facets.shape[1] - 1
    M = np.zeros((n_vert,n_facets),int)
    for i in range(n_facets):
        vert_set = facets[i]
        for v in vert_set:
            M[v, i] = 1
    L = FaceLattice(M, dim_V)
    return L
