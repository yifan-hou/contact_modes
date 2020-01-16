from __future__ import division
import numpy as np

from .polytope import Polytope
from .se3 import SE3
from .so3 import SO3


class Box(Polytope):
    def __init__(self, x=1.0, y=1.0, z=1.0):
        V = np.zeros((3,8))
        V[0,:] = 0.5*np.array([x, x, x, x, -x, -x, -x, -x])
        V[1,:] = 0.5*np.array([y, -y, y, -y, y, -y, y, -y])
        V[2,:] = 0.5*np.array([-z, -z, z, z, -z, -z, z, z])
        super(Box, self).__init__(V)
        self.x = x
        self.y = y
        self.z = z

        self.visual_box = Polytope(V)
        self.visual_box.subdivide_4_1()