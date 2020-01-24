from __future__ import division
import numpy as np

from .shape import Shape
from contact_modes import SE3, SO3

from contact_modes.viewer.backend import *


class Grid(Shape):
    def __init__(self, step, size):
        super().__init__()
        self.init_opengl(step, size)

    def init_opengl(self, step, size):
        # ----------------------------------------------------------------------
        # 1. Create data for grid
        # ----------------------------------------------------------------------
        steps = np.arange(step, size, step)
        n_pts = 4*2*len(steps) + 4
        vertices = np.zeros((3, n_pts), dtype='float32')
        normals = np.zeros((3, n_pts), dtype='float32')
        colors = np.ones((3, n_pts), dtype='float32')
        k = 0
        for i in steps:
            # lines parallel to x-axis
            vertices[:,k] = np.array([-size, i, 0])
            k += 1
            vertices[:,k] = np.array([ size, i, 0])
            k += 1
            vertices[:,k] = np.array([-size,-i, 0])
            k += 1
            vertices[:,k] = np.array([ size,-i, 0])
            k += 1
            # lines parallel to y-axis
            vertices[:,k] = np.array([ i, -size, 0])
            k += 1
            vertices[:,k] = np.array([ i,  size, 0])
            k += 1
            vertices[:,k] = np.array([-i, -size, 0])
            k += 1
            vertices[:,k] = np.array([-i,  size, 0])
            k += 1
        # x-axis
        vertices[:,k] = np.array([-size, 0, 0])
        colors[:,k] = np.array([1, 0, 0])
        k += 1
        vertices[:,k] = np.array([ size, 0, 0])
        colors[:,k] = np.array([1, 0, 0])
        k += 1
        # y-axis
        vertices[:,k] = np.array([0, -size, 0])
        colors[:,k] = np.array([0, 0, 1])
        k += 1
        vertices[:,k] = np.array([0,  size, 0])
        colors[:,k] = np.array([0, 0, 1])
        k += 1
        # normals point upwards
        normals[2,:] = 1.0
        # flatten
        vertices = vertices.T.flatten()
        normals = normals.T.flatten()
        colors = colors.T.flatten()

        # ----------------------------------------------------------------------
        # 2. Create VBOs
        # ----------------------------------------------------------------------
        self.num_elements_draw = len(vertices)
        self.vao = glGenVertexArrays(1)
        self.vertex_vbo, self.normal_vbo, self.color_vbo = glGenBuffers(3)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(vertices)*4, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self.normal_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(normals)*4, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(colors)*4, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def draw(self, shader):
        shader.use()
        model = self.get_tf_world().matrix().T
        shader.set_mat4('model', model)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, self.num_elements_draw)
        glBindVertexArray(0)
    