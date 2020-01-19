from contact_modes import get_data

from .backend import *
from .shader import *
from .quad import Quad

DEBUG = True

class BasicLightingRenderer(object):
    def __init__(self, window):
        self.window = window
        self.peel_depth = 16
        self.max_peel_depth = 64
        self.opacity = 0.6

    def set_draw_func(self, draw_func):
        self.draw_func = draw_func

    def init_opengl(self):
        pass

    def render(self):
        pass