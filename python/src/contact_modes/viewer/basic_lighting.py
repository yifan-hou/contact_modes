from contact_modes import get_data

from .backend import *
from .shader import *

DEBUG = True

class BasicLightingRenderer(object):
    def __init__(self, window):
        self.window = window
        self.peel_depth = 16
        self.max_peel_depth = 64
        self.opacity = 0.7

    def set_draw_func(self, draw_func):
        self.draw_func = draw_func

    def init_opengl(self):
        # Basic lighting shader.
        vertex_source = os.path.join(get_data(), 'shader', 'basic_lighting.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'basic_lighting.fs')
        self.basic_lighting_shader = Shader([vertex_source], [fragment_source])

    def render(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)

        self.draw_func(self.basic_lighting_shader)
        