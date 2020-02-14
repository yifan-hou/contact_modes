from time import time

import numpy as np

from contact_modes.geometry import *

np.set_printoptions(suppress=True, precision=5, linewidth=250, sign=' ')

def test_initial_arrangement():
    np.random.seed(0)
    for d in range(2, 9):
        A = np.random.normal(size=(d, d))
        for i in range(d):
            A[i,:] /= np.linalg.norm(A[i,:])
        V, S = zonotope_vertices(A)
        S = lexsort(S)
        # Create initial arrangement.
        b = np.zeros((d,1))
        I = initial_arrangement(A, b)
        assert(np.all(lexsort(I.sign_vectors(d)) == S))

def test_increment_arrangement_linear():
    np.random.seed(0)
    n = 10
    d = 4
    A = np.random.normal(size=(n, d))
    for i in range(d):
        A[i,:] /= np.linalg.norm(A[i,:])
    b = np.zeros((n,1))
    t_arrange = 0
    t = time()
    I = initial_arrangement(A[0:d], b[0:d])
    t_arrange += time() - t
    t_zono = 0
    for i in range(d, n):
        t = time()
        I0 = zonotope_incidence_graph_opposite(A[0:i+1])
        t_zono = time() - t
        t = time()
        increment_arrangement(A[i], b[i], I)
        t_arrange += time() - t
        for k in range(0,d):
            S0 = lexsort(I0.get_positions(k))
            S1 = lexsort(I.sign_vectors(k+1))
            assert(np.all(S1 == S0))
    print('t zono', t_zono)
    print('t incr', t_arrange)
    assert(False)