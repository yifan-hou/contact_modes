from time import time

import numpy as np
import sympy as sp

from contact_modes.shape import Chain


def test_chain():
    sp.init_printing()
    chain = Chain(5)
    chain.generate_forward_kinematics()
    t_start = time()
    for i in range(1000):
        chain.set_dofs(np.random.rand(8))
    print((time() - t_start)/1000)
    assert(False)
