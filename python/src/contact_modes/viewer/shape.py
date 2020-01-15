import numpy as np
from itbl.math.lie import SE3, sqrt_epsilon

class Shape(object):
    def __init__(self, o2w=None, w2o=None):
        if o2w is None:
            o2w = SE3.identity()
        if w2o is None:
            w2o = SE3.identity()
        self.o2w = o2w
        self.w2o = w2o
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_tf_world(self, tf_world):
        self.o2w = tf_world

    def get_tf_world(self):
        return self.o2w

    def object_bound(self):
        pass

    def world_bound(self):
        pass

    def is_convex(self):
        return False

    def margin(self):
        return sqrt_epsilon()

    def supmap(self, v, use_margin=False):
        return None, None