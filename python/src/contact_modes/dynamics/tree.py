import numpy as np

from contact_modes import SE3, SO3

from .body import *
from .link import *


class Tree(Body):
    def __init__(self):
        self.links = []

    def init(self):
        pass

    def num_dofs(self):
        return len(self.q)

    def get_dofs(self):
        return self.q

    def set_dofs(self, q):
        self.q = q.reshape((-1,1))
        for link in self.links:
            link.set_dofs(self.q)

    def draw(self, shader):
        for link in self.links:
            link.draw(shader)