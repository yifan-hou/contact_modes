import numpy as np

from contact_modes import SE3, SO3


class Body(object):
    def __init__(self):
        self.g_wb = SE3.identity()
        self.mask = np.array([True] * 6, bool)

    def get_transform_world(self):
        return self.g_wb
    
    def set_transform_world(self, g):
        self.g_wb = g
        if self.shape is not None:
            self.shape.set_tf_world(self.g_wb)

    def num_dofs(self):
        return np.sum(self.mask)

    def get_dofs(self):
        q = np.zeros((len(self.mask), 1))
        q[self.mask] = SE3.log(self.g_wb)
        return q

    def set_dofs(self, q):
        self.set_transform_world(SE3.exp(q[self.mask]))

    def get_shape(self):
        return self.shape

    def set_shape(self, shape):
        self.shape = shape

    def get_dof_mask(self):
        return self.mask

    def set_dof_mask(self, mask):
        self.mask = mask

    def get_body_jacobian(self):
        return SE3.identity()

    def get_spatial_jacobian(self):
        return SE3.Ad(self.get_transform_world()) @ self.get_body_jacobian()

    def draw(self, shader):
        self.shape.draw(shader)

    def supmap(self, v):
        return self.shape.supmap(v)
    
    def margin(self):
        return self.shape.margin()
