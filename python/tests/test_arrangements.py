from time import time

import numpy as np

from contact_modes.geometry import *

np.set_printoptions(suppress=True, precision=5, linewidth=250, sign=' ')

def test_initial_arrangement():
    A = np.eye(6)
    b = np.zeros((6,1))
    # A = np.random.rand(6,6)
    # b = np.random.rand(6,1)
    initial_arrangement(A, b)
    # assert(False)

def test_increment_arrangement():
    d = 8

    A = np.eye(d)
    b = np.zeros((d,1))
    # A = np.random.rand(6,6)
    # b = np.random.rand(6,1)

    t_start = time()
    I = initial_arrangement(A, b)
    print('initial', time() - t_start)

    # print(num_incidences_simple(d))
    # print(I.num_incidences())
    # for i in range(3, 15):
    #     print(i, num_incidences_simple(i))

    for i in range(3):
        a = np.random.rand(1,d)
        b = np.random.rand(1,1)
        t_start = time()
        increment_arrangement(a, b, I)
        print('increment', time() - t_start)

    assert(False)
