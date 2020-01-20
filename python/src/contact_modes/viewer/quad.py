import glm
import numpy as np

from .backend import *
from .shape import Shape


class Quad(Shape):
    def __init__(self):
        super().__init__()
    
    def init_opengl(self):
        data = np.array([-1.0,  1.0, 0.0, 0.0, 1.0,
                         -1.0, -1.0, 0.0, 0.0, 0.0,
                          1.0,  1.0, 0.0, 1.0, 1.0,
                          1.0, -1.0, 0.0, 1.0, 0.0], dtype='float32')
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(data)*4, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))

    def draw(self, shader):
        shader.use()
        identity = np.asarray(glm.mat4(1.0))
        shader.set_mat4('model', identity)
        shader.set_mat4('view', identity)
        shader.set_mat4('projection', identity)

        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
