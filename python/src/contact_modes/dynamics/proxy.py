import numpy as np

from contact_modes import SE3, SO3


class Proxy(Body):
    def __init__(self):
        self.body = None

    def set_body(self, body):
        self.body = body

    def get_body_jacobian(self):
        return self.body.get_body_jacobian()

    def get_spatial_jacobian(self):
        return self.body.get_spatial_jacobian()