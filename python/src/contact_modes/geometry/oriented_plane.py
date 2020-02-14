import numpy as np

from .incidence_graph import *
from .arrangements import *


def project_oriented_plane(A):
    n = A.shape[0]
    d = A.shape[1]
    A0 = A[:,0:d-1]
    b0 = A[:,d-1].reshape((-1,1))
    return A0, b0