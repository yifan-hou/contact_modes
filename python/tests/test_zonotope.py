from time import time
import numpy as np

from contact_modes.geometry import *

np.set_printoptions(suppress=True, precision=8, sign='+')

def test_zonotope():
    np.random.seed(0)
    n = 10
    d = 3
    A = np.random.normal(size=(n, d))
    for i in range(n):
        A[i,:] /= np.linalg.norm(A[i,:])
    V = np.array([A[0,:],-A[0,:]])
    S = np.array([[1], [-1]], int)
    for i in range(1, n):
        t = time()
        V, S = zonotope_minkowski_sum(A[i], V, S)
        t = time() - t
        print('%02d %0.6f' % (i, t), len(V), d**i)
        print(lexsort(S))
    assert(False)

def test_zonotope_incidence_graph():
    np.random.seed(0)
    n = 5
    d = 3
    A = np.random.normal(size=(n, d))
    for i in range(n):
        A[i,:] /= np.linalg.norm(A[i,:])
    I = zonotope_incidence_graph_opposite(A)
    for k in range(d):
        print(I.get_positions(k))
    assert(False)