import contact_modes
from contact_modes import inv_vert_2d
import numpy as np

np.set_printoptions(suppress=True, precision=5, linewidth=210)
np.random.seed(0)

def test_inv_vert_2d():
    mnp = []
    mnp.append((np.array([ 0, 1.]), np.array([0,-1.]), 0))
    env = []
    env.append((np.array([-1, 0.]), np.array([0, 1.]), 0))
    env.append((np.array([ 1, 0.]), np.array([0, 1.]), 0))
    v = np.array([1.0, 0.0, 0.0])
    x = np.array([0.0, 0.5, 0.0])

    inv_vert_2d(v, x, mnp, env)
    # assert(False)

test_inv_vert_2d()