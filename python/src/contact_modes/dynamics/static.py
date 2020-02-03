import numpy as np

from contact_modes import SE3, SO3

from .body import *


DEBUG=True

class Static(Body):
    def __init__(self):
        super(Static, self).__init__()
        self.g_wl0 = SE3.identity() # transform at q=0

    def set_transform_0(self, g_wl0):
        self.g_wl0 = g_wl0

    def get_dofs(self):
        return None

    def set_dofs(self, q):
        pass

    def get_body_jacobian(self):
        return None

    def get_spatial_jacobian(self):
        return None
