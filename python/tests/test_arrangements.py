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
    A0, b0 = project_oriented_plane(A)

    t_arrange = 0
    t_arrange0 = 0
    t_zono = 0

    t = time()
    I = initial_arrangement(A[0:d], b[0:d])
    t_arrange += time() - t

    t = time()
    I0 = initial_arrangement(A0[0:d-1], b0[0:d-1])
    increment_arrangement(A0[d-1], b0[d-1], I0)
    t_arrange0 += time() - t

    for i in range(d, n):
        t = time()
        Iz = zonotope_incidence_graph_opposite(A[0:i+1])
        t_zono = time() - t

        t = time()
        print('LINEAR')
        increment_arrangement(A[i], b[i], I)
        t_arrange += time() - t

        t = time()
        print('ORIENTED PLANE')
        increment_arrangement(A0[i], b0[i], I0)
        t_arrange0 += time() - t

        for k in range(0,d):
            S0 = lexsort(Iz.get_positions(k))
            S1 = lexsort(I.sign_vectors(k+1))
            assert(np.all(S1 == S0))
    print('t zono ', t_zono)
    print('t incr ', t_arrange)
    print('t incr0', t_arrange0)
    assert(False)