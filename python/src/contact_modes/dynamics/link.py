import numpy as np

from contact_modes import SE3, SO3

from .body import *


DEBUG=False

class Link(Body):
    def __init__(self, name=None):
        super(Link, self).__init__(name=name)
        self.q = None               # dofs
        self.xi = None              # joint twists
        self.g_wl0 = SE3.identity() # transform at q=0
        self.parent = None
        self.childs = []

    def add_child(self, link):
        self.childs.append(link)
        link.parent = self

    def set_transform_0(self, g_wl0):
        self.g_wl0 = g_wl0
    
    def get_transform_0(self):
        return self.g_wl0

    def set_joint_twists(self, xi):
        self.xi = xi

    def get_dofs(self):
        q = np.zeros((len(self.mask), 1))
        q[self.mask] = self.q
        return q

    def set_dofs(self, q):
        self.q = q[self.mask]
        exp = SE3.identity()
        for i in range(self.num_dofs()):
            q = self.q[i,0]
            xi = self.xi[i]
            exp = exp * SE3.exp(xi * q)
        g_wl = exp * self.g_wl0
        self.set_transform_world(g_wl)

    def get_body_jacobian(self):
        J_s = self.get_spatial_jacobian()
        J_b = SE3.Ad(SE3.inverse(self.get_transform_world())) @ J_s
        return J_b

    def get_spatial_jacobian(self):
        J = np.zeros((6, self.num_dofs()))
        exp = SE3.identity()
        for i in range(self.num_dofs()):
            if i-1 > 0:
                exp = exp * SE3.exp(self.xi[i-1] * self.q[i-1,0])
            J[:,i,None] = SE3.Ad(exp) @ self.xi[i]
        J_s = np.zeros((6, len(self.mask)))
        J_s[:,self.mask] = J
        return J_s
