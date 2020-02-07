import numpy as np

from contact_modes import SE3, SO3


class Body(object):
    def __init__(self, name=None):
        self.g_wb = SE3.identity()
        self.mask = np.array([True] * 6, bool)
        self.name = name
        self.contacts = []
        self.qdot = np.zeros((6,1))

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
        return 6

    def get_state(self):
        q = np.zeros((len(self.mask), 1))
        q[self.mask] = SE3.log(self.g_wb)
        return q

    def set_state(self, q):
        q = np.array(q).reshape((-1,1))
        self.set_transform_world(SE3.exp(q[self.mask]))

    def get_velocity(self):
        qdot = np.zeros((len(self.mask), 1))
        qdot[self.mask] = self.qdot
        return qdot

    def set_velocity(self, qdot):
        qdot = np.array(qdot).reshape((-1,1))
        self.qdot = qdot[self.mask]

    def step(self, q_dot):
        q_dot = np.array(q_dot).reshape((-1,1))
        self.set_velocity(q_dot)
        v_b = q_dot[self.mask]
        g_prev = self.get_transform_world()
        v_s = SE3.Ad(g_prev) @ v_b
        g_next = SE3.exp(v_s) * g_prev
        self.set_transform_world(g_next)

    def get_shape(self):
        return self.shape

    def set_shape(self, shape):
        self.shape = shape
        self.shape.set_tf_world(self.get_transform_world())

    def get_dof_mask(self):
        return self.mask

    def reindex_dof_mask(self, index, total_dofs):
        mask = np.array([False] * total_dofs, bool)
        mask[index:(index + len(self.mask))] = self.mask
        self.mask = mask

    def set_dof_mask(self, mask):
        self.mask = mask

    def get_body_jacobian(self):
        J_b = np.zeros((6, len(self.mask)))
        J_b[:, self.mask] = np.eye(6)
        return J_b

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
