import numpy as np

from contact_modes import SE3, SO3

from .body import *
from .link import *


class Tree(Body):
    def __init__(self):
        self.links = []

    def init(self):
        pass

    def num_links(self):
        return len(self.links)

    def get_links(self):
        return self.links

    def init_tree(self):
        for i in range(self.num_links()):
            l0 = self.links[i]
            m0 = l0.get_dof_mask()
            for j in range(i+1, self.num_links()):
                l1 = self.links[j]
                m1 = l1.get_dof_mask()
                if np.sum(np.logical_xor(m0, m1)) == 1:
                    idx = np.where(np.logical_xor(m0, m1))[0].item()
                    if m0[idx]:
                        l0.add_child(l1)
                    else:
                        l1.add_child(l0)

    def get_collision_flag(self):
        return self.flag

    def set_collision_flag(self, flag):
        self.flag = flag
        for link in self.links:
            link.set_collision_flag(flag)

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
    
    def set_color(self, color):
        for link in self.links:
            link.set_color(color)