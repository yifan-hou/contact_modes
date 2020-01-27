from __future__ import division

import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from contact_modes import SE3, SO3, get_data, get_color
from contact_modes.viewer.backend import *

from .collada import import_mesh
from .halfedgemesh import HalfedgeMesh

def smallfmod(x, y):
    while x > y:
        x -= y
    while x < 0:
        x += y
    return x

class Torus(HalfedgeMesh):
    def __init__(self, r1=1.0, r2=0.5, strip_color=get_color('clay')):
        super(Torus, self).__init__()
        self.strip_color = strip_color

        # Load torus.
        dae_path = os.path.join(get_data(), 'mesh', 'torus.dae')
        import_mesh(dae_path, self)

        # Compute toroidal coordinates.
        n_verts = len(self.vertices)
        P = np.zeros((3, n_verts))
        for i in range(n_verts):
            v = self.vertices[i]
            x, y, z = v.position
            theta = np.arctan2(y, x)[0]
            R = SO3.exp(np.array([0, 0, -theta]))
            P[:,i,None] = SO3.transform_point(R, v.position)
        center = np.mean(P, axis=1, keepdims=True)
        # radius = np.mean(norm(P - center, axis=0))

        # self.subdivide_4_1()

        # Rescale torus.
        n_verts = len(self.vertices)
        for i in range(n_verts):
            v = self.vertices[i]
            x, y, z = v.position
            theta = np.arctan2(y, x)[0]
            R = SO3.exp(np.array([0, 0, theta]))
            c0 = SO3.transform_point(R, center)
            o0 = v.position - c0
            c1 = r1 * c0/norm(c0)
            o1 = r2 * o0/norm(o0)
            v.position = c1 + o1

        # Rebuild torus.
        self.init()

    def init(self):
        self.reindex()
        self.set_tf_world(SE3.identity())

        self.init_geometry()
        self.init_opengl()

    def init_opengl(self, smooth=True):
        # Get per face vertex positions, normals, and colors.
        vertices = np.zeros((3, 3*len(self.faces)), dtype='float32')
        normals = np.zeros((3, 3*len(self.faces)), dtype='float32')
        colors = np.ones((4, 3*len(self.faces)), dtype='float32')
        k = 0
        for i in range(len(self.faces)):
            f = self.faces[i]
            assert(f.index == i)
            h = f.halfedge
            assert(k == 3*f.index)
            while True:
                v = h.vertex
                vertices[:,k,None] = v.position
                if not smooth:
                    normals[:,k,None]  = f.normal
                else:
                    normals[:,k,None]  = v.normal
                # 
                x, y, z = f.position
                theta = np.arctan2(y, x)[0]
                cut = np.pi/3
                tmod = smallfmod(theta, cut)
                if tmod > cut/2:
                    colors[:,k] = get_color('white')
                else:
                    colors[:,k] = self.strip_color
                # colors[:,k] = get_color('clay')
                # colors[0:3,k] = np.random.rand(3)
                h = h.next
                k += 1
                if h is f.halfedge:
                    break
        vertices = vertices.T.flatten()
        normals = normals.T.flatten()
        colors = colors.T.flatten()

        self.num_elems_draw = len(vertices)

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

    def set_color(self, color):
        for f in self.faces:
            x, y, z = f.position
            theta = np.arctan2(y, x)[0]
            cut = np.pi/3
            tmod = smallfmod(theta, cut)
            if tmod <= cut/2:
                f.set_color(color)