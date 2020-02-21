import numpy as np

from contact_modes.viewer.backend import *


class Body2D(object):
    def __init__(self, name=None):
        self.g_wb = np.zeros((3,1))
        self.name = name
        self.contacts = []
        self.shape = None
        self.c_shape = None

    def num_dofs(self):
        return 3

    def reset_contacts(self):
        self.contacts.clear()

    def set_pose(self, g):
        g = np.array(g).reshape((3,1))
        self.g_wb = g
        if self.shape is not None:
            self.shape.set_pose(g)
        if self.c_shape is not None:
            self.c_shape.set_pose(g)
    
    def get_pose(self):
        return self.g_wb
    
    def get_shape(self):
        return self.shape

    def set_shape(self, shape):
        self.shape = shape
        self.set_pose(self.get_pose())

    def get_collision_shape(self):
        return self.c_shape

    def set_collision_shape(self, shape):
        self.c_shape = shape
        self.c_shape.draw_filled = False
        self.c_shape.set_pose(self.get_pose())
    
    def draw(self, shader):
        self.shape.draw(shader)

        if self.c_shape is not None:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

            glEnable(GL_LINE_STIPPLE)

            glLineStipple(1, 0x0FFF)
            self.c_shape.draw(shader)

            glDisable(GL_LINE_STIPPLE)

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

class Static2D(Body2D):
    def __init__(self, name=None):
        super(Static2D, self).__init__(name)

    def num_dofs(self):
        return 0