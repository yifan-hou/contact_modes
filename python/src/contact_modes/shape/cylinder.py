import numpy as np

from contact_modes import SO3, SE3

from .halfedgemesh import HalfedgeMesh
from .polytope import Polytope


class Cylinder(HalfedgeMesh):
    def __init__(self, radius=1.0, height=1.0, n=30):
        super(Cylinder, self).__init__()
        self.radius = radius
        self.height = height

        V = np.zeros((3, 2*n), dtype='float32')
        for i in range(n):
            t = i/n*2*np.pi
            V[:,i] = np.array([radius*np.cos(t), radius*np.sin(t), -height/2.0])
        for i in range(n, 2*n):
            t = i/n*2*np.pi
            V[:,i] = np.array([radius*np.cos(t), radius*np.sin(t), height/2.0])
        self.build_convex(V)

    def supmap(self, v):
        m = self.margin()

        tf = self.get_tf_world()
        r = self.radius - m
        l = self.height - 2 * m

        v = SO3.transform_point_by_inverse(tf.R, v)
        w = v.copy()
        w[2,0] = 0
        x = v[2,0] * l/2 * np.array([0, 0, 1.0]).reshape((3,1))
        if np.linalg.norm(w) > 1e-6:
            x += r * w / np.linalg.norm(w)
        x = SE3.transform_point(tf, x)

        return x
