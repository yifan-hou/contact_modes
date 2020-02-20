import numpy as np
from scipy.spatial import ConvexHull

from contact_modes import SE3, get_color
from contact_modes.viewer.backend import *

from .halfedgemesh import HalfedgeMesh, reorient
from .shape_2d import Shape2D


def orient_ccw_2d(pts):
    c = np.mean(pts, axis=1)
    t = []
    for i in range(pts.shape[1]):
        d = pts[:,i] - c
        t.append(np.arctan2(d[1], d[0]))
    t = np.array(t)
    I = np.argsort(t)
    return pts[:, I]

class Polygon2D(Shape2D):
    def __init__(self, points):
        super(Polygon2D, self).__init__()

        self.build_convex(points)

    def num_vertices(self):
        return self.vertices.shape[1]

    def build_convex(self, points):
        hull = ConvexHull(points.T)
        self.vertices = orient_ccw_2d(hull.points.T)
        self.init_opengl()
    
    def init_opengl(self):
        # Get vertex positions, normals, and colors.
        n_verts = self.num_vertices()
        vertices = np.zeros((3, n_verts), dtype='float32')
        normals  = np.zeros((3, n_verts), dtype='float32')
        colors   = np.zeros((4, n_verts), dtype='float32')
        for i in range(n_verts):
            vertices[0:2,i] = self.vertices[:,i]
            normals[2,i] = 1.0
            colors[:,i] = get_color('clay')
        vertices = vertices.T.flatten()
        normals = normals.T.flatten()
        colors = colors.T.flatten()

        # Setup VAO.
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
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def draw(self, shader):
        model = SE3.exp([self.q[0], self.q[1], 0, 0, 0, self.q[2]])
        shader.set_mat4('model', model)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, self.num_vertices())
        glBindVertexArray(0)

class Box2D(Polygon2D):
    def __init__(self, width=1, height=1):
        points = np.array([[1, 1, -1, -1], [1, -1, 1, -1]], float)
        super(Box2D, self).__init__(points)

        self.width = width
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height
    
    def draw(self, shader):
        model = SE3.exp([self.q[0], self.q[1], 0, 0, 0, self.q[2]]).matrix()
        scale = np.diag([self.width/2, self.height/2, 1, 1])
        shader.set_mat4('model', model @ scale)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, self.num_vertices())
        glBindVertexArray(0)