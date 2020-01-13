from time import time
import numpy as np
import scipy as sp
import contact_modes as cm


def test_int_pt_cone():
    np.random.seed(0)
    C = np.random.rand(6,5)
    t_start = time()
    n = 100
    for i in range(100):
        pt = cm.int_pt_cone(C)
    print((time()-t_start)/n)

    b = np.zeros((C.shape[0], 1))
    t_start = time()
    for i in range(100):
        pt = cm.interior_point_halfspace(C, b)
    print((time()-t_start)/n)

    print(pt)
    assert(False)