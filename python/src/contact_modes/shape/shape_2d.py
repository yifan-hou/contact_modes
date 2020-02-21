import numpy as np


class Shape2D(object):
    def __init__(self):
        self.q = np.zeros((3,1))
        self.draw_filled  = True
        self.draw_outline = True

    def get_pose(self):
        return self.q

    def set_pose(self, q):
        self.q = q

    def set_color(self, color):
        pass

    def draw(self, shader):
        assert(False)