import numpy as np

from contact_modes import SE3, SO3

from .body import Body


class Proxy(Body):
    def __init__(self, name=None):
        super(Proxy, self).__init__(name)
        self.body = None
        self.g_bp = SE3.identity()
        self.m = 0
        self.n_dofs = 1

    def set_num_dofs(self, num_dofs):
        print('WARNING: Hacking proxy DOFs')
        self.n_dofs = num_dofs

    def num_dofs(self):
        return self.n_dofs

    def get_transform_world(self):
        return self.body.get_transform_world()
    
    def set_transform_body(self, g_bp):
        self.g_bp = g_bp

    def set_body(self, body):
        self.body = body

    def get_dof_mask(self):
        return self.body.get_dof_mask()

    def get_body_jacobian(self):
        return self.body.get_body_jacobian()

    def get_spatial_jacobian(self):
        return self.body.get_spatial_jacobian()

    def supmap(self, v):
        g_wb = self.get_transform_world()
        g_wp = g_wb * self.g_bp

        self.get_shape().set_tf_world(g_wp)

        return self.get_shape().supmap(v)

    def set_margin(self, m):
        self.m = m

    def margin(self):
        return self.m