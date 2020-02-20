import numpy as np


class Shape2D(object):
    def __init__(self):
        self.q = np.zeros((3,1))

    def get_state(self, q):
        return self.q

    def set_state(self, q):
        self.q = q

    def draw(self, shader):
        assert(False)