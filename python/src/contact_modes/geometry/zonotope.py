import numpy as np
import pyhull

from .affine import project_affine_subspace
from .incidence_graph import *

DEBUG=False

def zonotope_incidence_graph_opposite(A):
    V, S = zonotope_vertices(A)

    V_aff = project_affine_subspace(V.T).T
    if DEBUG:
        print('V_aff')
        print(V_aff)

    verts = [list(V_aff[i]) for i in range(V_aff.shape[0])]
    ret = pyhull.qconvex('Fv', verts)

    M = build_vertex_facet_matrix(ret, verts).T

    n = A.shape[0]
    d = V_aff.shape[1]
    I = build_incidence_graph(M, d)

    # Assign sign vectors for topes.
    for i in range(M.shape[1]):
        vf_key = tuple(np.where(M[:,i])[0].astype(int))
        f = I.rank(d-1)[vf_key]
        f.pos = S[i]
    # Assign sign vectors for k faces.
    for k in range(d-2, -1, -1):
        for u in I.rank(k).values():
            # print(u._sv_key, k)
            pos = np.zeros((n,), int)
            for v in u.superfaces:
                # print('v', v.pos)
                pos += v.pos
            pos = (pos / len(u.superfaces)).astype(int)
            u.pos = pos
            # print('u', u.pos)
    
    return I

def zonotope_vertices(A):
    n = A.shape[0]
    V = np.array([A[0,:],-A[0,:]])
    S = np.array([[1], [-1]], int)
    for i in range(1, n):
        V, S = zonotope_minkowski_sum(A[i], V, S)
    return V, S

def zonotope_minkowski_sum(a, V, S):
    """Compute the Minkowski sum V ⊕ [a,-a].
    
    Arguments:
        a {1xd vector} -- End-point of line segment [a,-a]
        V {nxd matrix} -- Zonotope vertices
        S {nxn matrix} -- Zonotope sign vectors
    
    Returns:
        [matrix, matrix] -- Updated zonotope vertices and sign vectors.
    """
    a = np.reshape(a, (1,-1))

    V_sum = np.vstack((V + a, V - a))
    if DEBUG:
        print('V_sum')
        print(V_sum)

    S_sum = np.ones((2*S.shape[0], S.shape[1]+1),dtype=int)
    S_sum[:,0:S.shape[1]] = np.vstack((S,S))
    S_sum[S.shape[0]:,-1] = -1
    if DEBUG:
        print('S_sum')
        print(S_sum)

    V_aff = project_affine_subspace(V_sum.T).T
    if DEBUG:
        print('V_aff')
        print(V_aff)

    vertices = [list(V_aff[i]) for i in range(V_aff.shape[0])]
    ret = pyhull.qconvex('Fv', vertices)

    idx = []
    for i in range(1, len(ret)):
        idx = idx + [int(x) for x in ret[i].split(' ')][1:]
    idx = np.unique(idx)
    if DEBUG:
        print('idx')
        print(idx)

    V = V_sum[idx]
    S = S_sum[idx]

    return V, S