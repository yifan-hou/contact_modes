import numpy as np

from contact_modes import SE3, SO3


class Body(object):
    def __init__(self, name=None):
        self.g_wb = SE3.identity()
        self.mask = np.array([True] * 6, bool)
        self.name = name
        self.contacts = []

    def reset_contacts(self):
        self.contacts.clear()

    def get_contacts(self):
        return self.contacts
    
    def add_contact(self, manifold):
        self.contacts.append(manifold)

    def get_collision_flag(self):
        return self.flag

    def set_collision_flag(self, flag):
        self.flag = flag

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
        self.shape.set_tf_world(self.get_transform_world())

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

    def set_color(self, color):
        self.shape.set_color(color)

    def supmap(self, v):
        return self.shape.supmap(v)
    
    def margin(self):
        return self.shape.margin()
