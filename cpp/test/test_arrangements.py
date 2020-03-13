from time import time

import numpy as np

import contact_modes._contact_modes as _cm
from contact_modes.geometry import *

np.set_printoptions(suppress=True, precision=5, linewidth=250, sign=' ')


def test_initial_arrangement():
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

