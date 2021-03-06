from time import time

import numpy as np

import contact_modes as cm
from contact_modes import FaceLattice


def test_face_lattice():
    # Square.
    M = np.array([[1, 0, 0, 1],
                  [1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1]])
    d = 2
    L = FaceLattice(M, d)
    # print(L.csmodes())

    # Cube.
    M = np.array([[1, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0],
                  [1, 0, 1, 1, 0, 0],
                  [1, 0, 0, 1, 1, 0],
                  [0, 1, 0, 0, 1, 1],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 1, 1, 1]])
    d = 3
    L = FaceLattice(M, d)

    # Box against wall.
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [1, 1, 0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 1, 1, 0, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1, 0]])
    d = 5
    L = FaceLattice(M, d)

    # Box against wall - Polar.
    M = np.array([[1, 1, 0, 1, 1, 0],
                  [1, 1, 0, 1, 0, 1],
                  [0, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 0],
                  [1, 0, 0, 1, 1, 1],
                  [1, 1, 0, 0, 1, 1],
                  [0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 1, 1, 1]])
    d = 4
    L = FaceLattice(M, d)

    # print(L.num_proper_faces())
    L.build_fast(M, d)
    # print(L.num_proper_faces())

    # t_start = time()
    # for i in range(1000):
    #     L.build(M, d)
    # print(time() - t_start)

    # t_start = time()
    # for i in range(1000):
    #     L.build_fast(M, d)
    # print(time() - t_start)

    # print(L.num_proper_faces())
    # print(L.num_faces())
    # print(L.csmodes())

    assert(False)


def test_hesse_diagram():
    return
    # Cube.
    M = np.array([[1, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0],
                  [1, 0, 1, 1, 0, 0],
                  [1, 0, 0, 1, 1, 0],
                  [0, 1, 0, 0, 1, 1],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 1, 1, 1]])
    d = 3
    L = FaceLattice(M, d)
    L.hesse_diagram('/tmp/cube.gv')

    # Box against wall - Polar.
    M = np.array([[1, 1, 0, 1, 1, 0],
                  [1, 1, 0, 1, 0, 1],
                  [0, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 0],
                  [1, 0, 0, 1, 1, 1],
                  [1, 1, 0, 0, 1, 1],
                  [0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 1, 1, 1]])
    d = 4
    L = FaceLattice(M, d)
    L.hesse_diagram('/tmp/wallbox_polar.gv')

    # assert(False)
