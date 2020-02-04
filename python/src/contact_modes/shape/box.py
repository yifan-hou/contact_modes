from __future__ import division
import numpy as np

from .halfedgemesh import HalfedgeMesh
from .bounds import Bounds3
from contact_modes import SE3, SO3


class Box(HalfedgeMesh):
    def __init__(self, x=1.0, y=1.0, z=1.0):
        super(Box, self).__init__()
        # Create vertices.
        self.x = x
        self.y = y
        self.z = z
        # x = x - 2 * self.margin()
        # y = y - 2 * self.margin()
        # z = z - 2 * self.margin()
        V = np.zeros((3,8))
        V[0,:] = 0.5*np.array([x, x, x, x, -x, -x, -x, -x])
        V[1,:] = 0.5*np.array([y, -y, y, -y, y, -y, y, -y])
        V[2,:] = 0.5*np.array([-z, -z, z, z, -z, -z, z, z])
        self.build_convex(V)
    
    def margin(self):
        return 0.01

    def supmap(self, v):
        tf = self.get_tf_world()

        x0 = self.x - 2 * self.margin()
        y0 = self.y - 2 * self.margin()
        z0 = self.z - 2 * self.margin()

        # x_max_dot = -np.Inf
        # x_max = np.zeros((3,1))
        # for x in [-x0/2, x0/2]:
        #     for y in [-y0/2, y0/2]:
        #         for z in [-z0/2, z0/2]:
        #             pt = np.array([x, y, z])
        #             pt = SE3.transform_point(tf, pt.reshape((3,1)))
        #             if x_max_dot < np.dot(v.T, pt).item():
        #                 x_max_dot = np.dot(v.T, pt).item()
        #                 x_max = pt

        v = SO3.transform_point_by_inverse(tf.R, v)
        x = np.multiply(np.sign(v), np.array([x0/2, y0/2, z0/2]).reshape((3,1)))
        x = SE3.transform_point(tf, x)

        # v = SO3.transform_point(tf.R, v)
        # x_dot = (v.T @ x).item()
        # assert(np.linalg.norm(x_dot - x_max_dot) < 1e-9)

        return x