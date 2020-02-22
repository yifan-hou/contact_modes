import glm
import math
import numpy as np


class OrthoCamera(object):
    def __init__(self, aspect_ratio=800./600.):
        self.aspect_ratio = aspect_ratio
        self.height = 2.0
        self.center = np.array([0.0, 0.0])
        self.last_x = 0.0
        self.last_y = 0.0

    def resize(self):
        pass

    def set_center():
        pass

    def get_view(self):
        return glm.mat4(1.0)

    def get_projection(self):
        x = self.center[0]
        y = self.center[1]
        h = self.height
        ar = self.aspect_ratio
        ortho = glm.ortho(x-h*ar, x+h*ar, y-h, y+h, -100.0, 100.0)
        return ortho

    def mouse_roll(self, norm_mouse_x, norm_mouse_y, dragging=True):
        if dragging:
            dx = norm_mouse_x - self.last_x
            dy = norm_mouse_y - self.last_y
            height = self.height
            width = self.aspect_ratio * self.height
            self.center[0] -= dx * width
            self.center[1] -= dy * height
        self.last_x = norm_mouse_x
        self.last_y = norm_mouse_y

    def mouse_zoom(self, norm_mouse_x, norm_mouse_y, dragging=True):
        if dragging:
            dx = norm_mouse_x - self.last_x
            dx = 0
            dy = norm_mouse_y - self.last_y
            norm_mouse_r_delta = 2.0*math.sqrt(dx*dx+dy*dy)
            if dy > 0.0:
                norm_mouse_r_delta = -norm_mouse_r_delta
            self.height = self.height * (1 + norm_mouse_r_delta)
        self.last_x = norm_mouse_x
        self.last_y = norm_mouse_y