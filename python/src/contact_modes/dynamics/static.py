import numpy as np

from contact_modes import SE3, SO3

from .body import *


DEBUG=True

class Static(Body):
    def __init__(self, name=None):
        super(Static, self).__init__(name=name)

    def num_dofs(self):
        return 0

    def get_dofs(self):
        return None

    def set_dofs(self, q):
        pass

    def get_body_jacobian(self):
        return None

    def get_spatial_jacobian(self):
        return None
