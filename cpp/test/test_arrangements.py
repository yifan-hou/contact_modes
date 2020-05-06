#!/usr/bin/env python3
from time import time

import numpy as np

import contact_modes._contact_modes as _cm
from contact_modes.geometry import *

np.set_printoptions(suppress=True, precision=5, linewidth=250, sign=' ')


def test_initial_arrangement():
    return
    np.random.seed(0)
    for d in range(2, 4):
        print('d', d)
        A = np.random.normal(size=(d, d))
        for i in range(d):
            A[i,:] /= np.linalg.norm(A[i,:])
        t_start = time()
        V, S = zonotope_vertices(A)
        print('zono: %.8f' % (time() - t_start))
        S = lexsort(S)
        # Create initial arrangement.
        b = np.zeros((d,1))
        t_start = time()
        I = initial_arrangement(A, b)
        print('arpy: %.8f' % (time() - t_start))
        t_start = time()
        I0 = _cm.initial_arrangement(A, b, 1e-8)
        print('ar++: %.8f' % (time() - t_start))
        I0.update_sign_vectors(1e-8)
        print(I0.get_sign_vectors())
        assert(np.all(lexsort(I.sign_vectors(d)) == S))
    assert(False)

def test_build_arrangement():
    # np.random.seed(0)

    # Assert each rank's positions match the python implementation (both
    # zonotope algorithm and hyperplane arrangement).
    for d in range(2, 6):
        for n in range(d, d + 4):
            print('d', d)
            print('n', n)
            # Uniformly sample linear hyperplanes.
            A = np.random.normal(size=(n, d))
            for i in range(n):
                A[i,:] /= np.linalg.norm(A[i,:])

            # Zonotope algorithm.
            t_start = time()
            V, S = zonotope_vertices(A)
            Iz = zonotope_incidence_graph_opposite(A)
            print('zono: %.8f' % (time() - t_start))
            S = lexsort(S)
            # print(S)

            # Python hyperplane arrangement.
            t_start = time()
            b = np.zeros((n,1))
            I = initial_arrangement(A[0:d], b[0:d])
            for i in range(d, n):
                increment_arrangement(A[i], b[i], I)
            print('arpy: %.8f' % (time() - t_start))

            # C++ hyperplane arrangement.
            t_start = time()
            I0 = _cm.build_arrangement(A, b, 1e-8)
            print('ar++: %.8f' % (time() - t_start))
            I0.update_sign_vectors(1e-8)

            # Assert positions match.
            for k in range(d+1):
                # print(lexsort(Iz.get_positions(k)))
                p0 = lexsort(np.array(I0.positions(k)))
                p = lexsort(I.sign_vectors(k))
                assert(np.all(p0 == p))
            # print(lexsort(np.array(I0.positions())))
            # print(I0.sign_vectors())
            # print(lexsort(I0.sign_vectors()))
            # assert(np.all(lexsort(I.sign_vectors(d)) == S))
    assert(False)

test_build_arrangement()