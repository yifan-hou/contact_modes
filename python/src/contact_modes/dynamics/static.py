import numpy as np

from contact_modes import SE3, SO3

from .body import *


DEBUG = False

class Static(Body):
    def __init__(self, name=None):
        super(Static, self).__init__(name=name)
        self.parent = None
        self.mask = np.array([], bool)

    def num_dofs(self):
        return 0

    def get_state(self):
        q = np.zeros((len(self.mask), 1))
        return q

    def set_state(self, q):
        pass

    def step(self, q_dot):
        pass

    def get_body_jacobian(self):
        return None

    def get_spatial_jacobian(self):
        return None

    def add_child(self, child):
        pass