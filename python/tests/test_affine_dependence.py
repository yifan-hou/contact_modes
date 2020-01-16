import numpy as np

from scipy.linalg import null_space as null
from scipy.linalg import orth

import contact_modes as cm


def test_61a():
    X = np.array([[0, 3, 5, 5, 2, 0], [0, 0, 1, 2, 2, 1]], dtype=float)

    print(cm.affine_dep(X))

    # assert(False)

def test_63():
    X = np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0],[0,0,0,0,1,-1]], float)
    print(cm.affine_dep(X))

    assert(False)