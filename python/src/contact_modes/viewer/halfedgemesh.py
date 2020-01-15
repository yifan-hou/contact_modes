# -*- coding: utf-8 -*-
import os
from time import time

import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull

from .backend import *
from .bounds import Bounds3, Point, union_bb, union_bp
from .se3 import SE3
from .shape import Shape
from .so3 import SO3


def area_triangle_3d(p0, p1, p2):
    u = p1 - p0
    v = p2 - p0
    return 0.5 * norm(cross(v, u))

def cross(x, y):
    return np.cross(x.reshape((3,1)), y.reshape((3,1)), axis=0)

def reorient(simplices, neighbors, points):
    # Orient the first simplex so that it is counter-clockwise.
    s = simplices[0]
    p = np.zeros((3,1))
    n = np.zeros((3,1))
    for i in range(len(s)):
        i1 = (i + 1) % len(s)
        v0 = points[s[i]].reshape((3,1))
        v1 = points[s[i1]].reshape((3,1))
        n += cross(v0, v1)
        p += v0
    n /= float(len(s))
    p /= float(len(s))
    if np.dot(n.T, p) < 0:
        s.reverse()
    # Reorient simplicies such that they are consistent with the first simplex.
    Q = []
    D = dict()
    H = dict()
    Q.append((0, simplices[0]))
    D[0] = 1
    while len(Q) > 0:
        i, s = Q.pop(0)
        degree = len(s)
        reverse = False
        for j in range(degree):
            a = s[j]
            b = s[(j+1)%degree]
            ab = (a,b)
            if ab in H.keys():
                reverse = True
        if reverse:
            s.reverse()
        for j in range(degree):
            a = s[j]
            b = s[(j+1)%degree]
            ab = (a,b)
            H[ab] = True
        for n in neighbors[i]:
            if n not in D.keys():
                Q.append((n, simplices[n]))
                D[n] = True
    return simplices

class Halfedge(Shape):
    def __init__(self):
        self.next = None
        self.twin = None
        self.vertex = None
        self.edge = None
        self.face = None
        self.index = None
        self.boundary = False

    def set_neighbors(self, n, t, v, e, f):
        self.next = n
        self.twin = t
        self.vertex = v
        self.edge = e
        self.face = f

    def on_boundary(self):
        return self.face.boundary

    def vector(self):
        return self.twin.vertex.position - self.vertex.position

class Edge(Shape):
    def __init__(self):
        self.halfedge = None
        self.index = None

    def adjacent_faces(self):
        """Convenience function to iterate over the faces adjacent to this edge."""
        return EdgeFaceIterator(self.halfedge)

class EdgeFaceIterator:
    """This class represents an adjacent face iterator for an edge."""

    def __init__(self, halfedge):
        if halfedge.onBoundary:
            halfedge = halfedge.twin
        self.__halfedge = halfedge

    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        if not self.justStarted and self.current == self.end:
            raise StopIteration
        self.justStarted = False
        face = self.current.face
        self.current = self.current.twin
        return face

    next = __next__

class Vertex(Shape):
    def __init__(self):
        self.halfedge = None
        self.position = np.zeros((3,1))
        self.world_position = np.zeros((3,1))
        self.index = None
        self.normal = np.zeros((3,1))

    def degree(self):
        """Counts the number of edges adjacent to this vertex."""
        k = 0
        for _ in self.adjacent_edges():
            k += 1
        return k

    def on_boundary(self):
        """Checks whether this vertex lies on a boundary."""
        for h in self.adjacent_halfedges():
            if h.on_boundary:
                return True
        return False

    def normal_area_weighted(self):
        N = np.zeros((3,1))
        for f in self.adjacent_faces():
            N += f.face_normal() * f.area()
        return N / norm(N)

    def adjacent_vertices(self):
        """Convenience function to iterate over the vertices neighboring this vertex."""
        return VertexVertexIterator(self.halfedge)

    def adjacent_edges(self):
        """Convenience function to iterate over the edges adjacent to this vertex."""
        return VertexEdgeIterator(self.halfedge)

    def adjacent_faces(self):
        """Convenience function to iterate over the faces adjacent to this vertex."""
        return VertexFaceIterator(self.halfedge)

    def adjacent_halfedges(self):
        """Convenience function to iterate over the halfedges adjacent to this vertex."""
        return VertexHalfedgeIterator(self.halfedge)

    def adjacent_corners(self):
        """Convenience function to iterate over the corners adjacent to this vertex."""
        return VertexCornerIterator(self.halfedge)

class VertexVertexIterator:
    """This class represents an adjacent vertex iterator for a vertex."""

    def __init__(self, halfedge):
        self.__halfedge = halfedge
    
    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        if not self.justStarted and self.current == self.end:
            raise StopIteration
        self.justStarted = False
        vertex = self.current.twin.vertex
        self.current = self.current.twin.next
        return vertex

    next = __next__

class VertexEdgeIterator:
    """This class represents an adjacent edge iterator for a vertex."""
    def __init__(self, halfedge):
        self.__halfedge = halfedge

    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        if not self.justStarted and self.current == self.end:
            raise StopIteration
        self.justStarted = False
        edge = self.current.edge
        self.current = self.current.twin.next
        return edge

    next = __next__

class VertexFaceIterator:
    """This class represents an adjacent face iterator for a vertex."""

    def __init__(self, halfedge):
        while(halfedge.on_boundary()):
            halfedge = halfedge.twin.next
        self.__halfedge = halfedge

    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        # halfedge must not be on the boundary
        while (self.current.on_boundary()):
            self.current = self.current.twin.next
        if not self.justStarted and self.current == self.end:
            raise StopIteration
        self.justStarted = False
        face = self.current.face
        self.current = self.current.twin.next
        return face

    next = __next__

class VertexHalfedgeIterator:
    """This class represents an adjacent halfedge iterator for a vertex."""

    def __init__(self, halfedge):
        self.__halfedge = halfedge
    
    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        if not self.justStarted and self.current == self.end:
            raise StopIteration
        self.justStarted = False
        halfedge = self.current
        self.current = self.current.twin.next
        return halfedge

    next = __next__

class VertexCornerIterator:
    """This class represents an adjacent corner iterator for a vertex."""

    def __init__(self, halfedge):
        while(halfedge.onBoundary):
            halfedge = halfedge.twin.next
        self.__halfedge = halfedge
    
    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        # halfedge must not be on the boundary
        while (self.current.onBoundary):
            self.current = self.current.twin.next
        if not self.justStarted and self.current == self.end:
            raise StopIteration
        self.justStarted = False
        corner = self.current.next.corner
        self.current = self.current.twin.next
        return corner

    next = __next__

class Face(Shape):
    def __init__(self, mesh=None, boundary=False):
        self.mesh = mesh
        self.halfedge = None
        self.boundary = boundary
        self.index = None
        self.normal = np.zeros((3,1))
        self.world_normal = np.zeros((3,1))
        self.area = 0

    def supmap(self, v, use_margin=False):
        x = np.zeros((3,1))
        x_max = -np.inf
        vert = None
        for w in self.adjacent_vertices():
            if x_max < np.dot(v.T, w.position):
                x_max = np.dot(v.T, w.position)
                x = w.position
                vert = w
        return x, vert

    def degree(self):
        n = 0
        h = self.halfedge
        while True:
            h = h.next
            n += 1
            if h is self.halfedge:
                break
        return n

    def object_bound(self):
        b = Bounds3()
        for v in self.adjacent_vertices():
            b = union_bp(b, v.position)
        return b

    def world_bound(self):
        b = Bounds3()
        for v in self.adjacent_vertices():
            b = union_bp(b, v.world_position)
        return b

    def set_color(self, color):
        color = color.astype('float32')
        glBindVertexArray(self.mesh.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh.color_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, (3*self.index+0)*3*4, 3*4, color)
        glBufferSubData(GL_ARRAY_BUFFER, (3*self.index+1)*3*4, 3*4, color)
        glBufferSubData(GL_ARRAY_BUFFER, (3*self.index+2)*3*4, 3*4, color)
        glBindVertexArray(0)

    def intersect_ray_p(self, ray):
        # Translate
        p0 = self.halfedge.vertex.position
        p1 = self.halfedge.next.vertex.position
        p2 = self.halfedge.next.next.vertex.position
        p0t = p0 - ray.o
        p1t = p1 - ray.o
        p2t = p2 - ray.o
        # Permute
        kz = np.argmax(np.abs(ray.d))
        kx = kz + 1
        if kx == 3:
            kx = 0
        ky = kx + 1
        if ky == 3:
            ky = 0
        d = ray.d[[kx, ky, kz]]
        p0t = p0t[[kx, ky, kz]]
        p1t = p1t[[kx, ky, kz]]
        p2t = p2t[[kx, ky, kz]]
        # Shear
        Sx = -d[0] / d[2]
        Sy = -d[1] / d[2]
        Sz = 1.0 / d[2]
        p0t[0] += Sx * p0t[2]
        p0t[1] += Sy * p0t[2]
        p1t[0] += Sx * p1t[2]
        p1t[1] += Sy * p1t[2]
        p2t[0] += Sx * p2t[2]
        p2t[1] += Sy * p2t[2]
        # Edge function
        e0 = p1t[0]*p2t[1] - p1t[1]*p2t[0]
        e1 = p2t[0]*p0t[1] - p2t[1]*p0t[0]
        e2 = p0t[0]*p1t[1] - p0t[1]*p1t[0]
        # Edge and determinant test
        if ((e0 < 0 or e1 < 0 or e2 < 0) and (e0 > 0 or e1 > 0 or e2 > 0)):
            return False
        det = e0 + e1 + e2
        if det == 0:
            return False
        # Scaled hit distance and range test
        p0t[2] *= Sz
        p1t[2] *= Sz
        p2t[2] *= Sz
        t_scaled = e0 * p0t[2] + e1 * p1t[2] + e2 * p2t[2]
        if (det < 0 and (t_scaled >= 0 or t_scaled < ray.t_max * det)):
            return False
        elif (det > 0 and (t_scaled <= 0 or t_scaled > ray.t_max * det)):
            return False
        # Barycentric coordinates and t value for triangle intersection
        inv_det = 1.0 / det
        b0 = e0 * inv_det
        b1 = e1 * inv_det
        b2 = e2 * inv_det
        t = t_scaled * inv_det

        # Ensure that computed triangle $t$ is conservatively greater than zero
        # Compute $\delta_z$ term for triangle $t$ error bounds
        maxZt = np.max(np.abs(np.array([p0t[2], p1t[2], p2t[2]])))
        deltaZ = gamma(3) * maxZt

        # Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
        maxXt = np.max(np.abs(np.array([p0t[0], p1t[0], p2t[0]])))
        maxYt = np.max(np.abs(np.array([p0t[1], p1t[1], p2t[1]])))
        deltaX = gamma(5) * (maxXt + maxZt)
        deltaY = gamma(5) * (maxYt + maxZt)

        # Compute $\delta_e$ term for triangle $t$ error bounds
        deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt)

        # Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
        maxE = np.max(np.abs(np.array([e0, e1, e2])))
        deltaT = 3 * (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * np.abs(inv_det)
        if (t <= deltaT):
            return False

        return True

    def adjacent_vertices(self):
        """Convenience function to iterate over the vertices neighboring this face."""
        return FaceVertexIterator(self.halfedge)

    def adjacent_faces(self):
        """Convenience function to iterate over the faces neighboring this face."""
        return FaceFaceIterator(self.halfedge)

class FaceVertexIterator:
    """This class represents an adjacent vertex iterator for a face."""

    def __init__(self, halfedge):
        self.__halfedge = halfedge

    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        if not self.justStarted and self.current is self.end:
            raise StopIteration
        self.justStarted = False
        vertex = self.current.vertex
        self.current = self.current.next
        return vertex

    next = __next__

class FaceFaceIterator:
    """This class represents an adjacent face iterator for a face."""

    def __init__(self, halfedge):
        self.__halfedge = halfedge

    def __iter__(self):
        self.current = self.__halfedge
        self.end = self.__halfedge
        self.justStarted = True
        return self

    def __next__(self):
        if not self.justStarted and self.current == self.end:
            raise StopIteration
        self.justStarted = False
        face = self.current.twin.face
        self.current = self.current.next
        return face

    next = __next__

class HalfedgeMesh(Shape):
    def __init__(self, o2w=None, w2o=None):
        super(HalfedgeMesh, self).__init__(o2w, w2o)
        self.halfedges = []
        self.vertices = []
        self.edges = []
        self.faces = []
        self.boundaries = []
        # OpenGL vertex arrays.
        self.vao = None
        self.vbo = None
        self.ebo = None

    def copy(self):
        mesh = HalfedgeMesh()
        mesh.build(self.simplices(), self.vertex_positions())
        return mesh

    def set_tf_world(self, tf):
        self.o2w = tf
        # Cache new vertex positions and face normals.
        for v in self.vertices:
            v.world_position = self.position(v)
        for f in self.faces:
            f.world_normal = self.normal(f)

    def new_halfedge(self):
        h = Halfedge()
        self.halfedges.append(h)
        return h

    def new_vertex(self):
        v = Vertex()
        self.vertices.append(v)
        return v
    
    def new_edge(self):
        e = Edge()
        self.edges.append(e)
        return e

    def new_face(self):
        f = Face(self)
        self.faces.append(f)
        return f

    def new_boundary(self):
        b = Face(boundary=True)
        self.boundaries.append(b)
        return b

    def world_bound(self):
        b = Bounds3()
        for f in self.faces:
            b = union_bb(b, f.world_bound())
        return b

    def surface_area(self):
        sa = 0
        for f in self.faces:
            sa += f.area
        return sa

    def area(self, f):
        u = -f.halfedge.vector()
        v =  f.halfedge.next.vector()
        return 0.5 * norm(cross(v, u))

    def barycentric_dual_area(self, v):
        total = 0
        for f in v.adjacent_faces():
            total += f.area
        return total / 3.0

    def dihedral_angle(self, h):
        pass

    def angle_defect(self, v):
        h = v.halfedge
        n = np.zeros((3,1), dtype='float32')
        total = 0
        while True:
            h0 = h
            h1 = h.twin
            h2 = h.twin.next
            h3 = h.twin.next.twin
            v0 = h0.vertex.position
            v1 = h1.vertex.position
            v2 = h3.vertex.position
            e0 = unit(v1-v0)
            e1 = unit(v2-v0)
            theta = np.arccos(dot(e0.T, e1))
            f = h1.face
            n = n + f.normal * theta
            h = h2
            if h is v.halfedge:
                break
        return 2*np.pi - total

    def scalar_gauss_curvature(self, v):
        return self.angle_defect(v)

    def scalar_mean_curvature(self, v):
        h = v.halfedge
        total = 0
        while True:
            e = h.vector()
            theta = self.dihedral_angle(h)
            sum += np.linalg.norm(e) * theta
            h = h.twin.next
            if h is v.halfedge:
                break
        return total/2.0

    def principal_curvatures(self, v):
        pass

    def position(self, v):
        return SE3.transform_point(self.get_tf_world(), v.position)

    def face_position(self, f):
        p = np.zeros((3,1))
        n = 0.0
        for v in f.adjacent_vertices():
            p += self.position(v)
            n += 1.0
        return p / n

    def vertex_positions(self):
        n = len(self.vertices)
        V = np.zeros((3,n))
        tf = self.get_tf_world()
        for i, v in zip(range(n), self.vertices):
            V[:,i,None] = SE3.transform_point(tf, v.position)
        return V

    def normal(self, f):
        N = np.zeros((3,1))
        h = f.halfedge
        while True:
            pi = h.vertex.position
            pj = h.next.vertex.position
            N += np.cross(pi, pj, axis=0)
            h = h.next
            if h is f.halfedge:
                break
        N = N / np.linalg.norm(N)
        N = SO3.transform_point(self.get_tf_world().R, N)
        return N

    def face_normals(self):
        n_f = len(self.faces)
        N = np.zeros((3,n_f))
        for i in range(n_f):
            N[:,i,None] = self.normal(self.faces[i])
        return N

    def vector(self, e):
        h0 = e.halfedge
        h1 = h0.twin
        v0 = h0.vertex.position
        v1 = h1.vertex.position
        v = SO3.transform_point(self.get_tf_world().R, v1-v0)
        return v

    def edge_vectors(self):
        n_e = len(self.edges)
        V = np.zeros((3,n_e))
        for i in range(n_e):
            V[:,i,None] = self.vector(self.edges[i])
        return V

    def reindex(self):
        for h, i in zip(self.halfedges, range(len(self.halfedges))):
            h.index = i
        for v, i in zip(self.vertices, range(len(self.vertices))):
            v.index = i
        for e, i in zip(self.edges, range(len(self.edges))):
            e.index = i
        for f, i in zip(self.faces, range(len(self.faces))):
            f.index = i

    def simplices(self):
        self.reindex()
        simplices = []
        for f in self.faces:
            s = []
            for v in f.adjacent_vertices():
                s.append(v.index)
            simplices.append(s)
        return simplices

    def supmap(self, v, use_margin=False):
        """Returns the support mapping, i.e. argmax v⋅x, x ∈ g⋅V.
        
        Arguments:
            v {np.ndarray} -- Input vector (3x1).
        
        Keyword Arguments:
            use_margin {bool} -- Use the objects margin for V. (default: {False})
        
        Returns:
            tuple -- (x, vertex)
        """
        g = self.o2w
        x_max_dot = -np.Inf
        x_max = np.zeros((3,1))
        vert = self.vertices[0]
        while True:
            adj_closer = False
            for v_adj in vert.adjacent_vertices():
                x = SE3.transform_point(g, v_adj.position)
                if x_max_dot < np.dot(v.T,x).item():
                    x_max_dot = np.dot(v.T,x).item()
                    x_max = x
                    adj_closer = True
                    vert = v_adj
            if not adj_closer:
                return x_max, vert

    def object_bound(self):
        b = Bounds3()
        for v in self.vertices:
            b = union_bp(b, v.position)
        return b

    def init(self):
        self.reindex()
        self.set_tf_world(SE3.identity())

        self.init_geometry()
        self.init_opengl()

    def init_geometry(self):
        # Compute face normals and areas.
        for f in self.faces:
            f.normal = self.normal(f)
            f.area = self.area(f)
            f.position = self.face_position(f)
        # Compute area-weighted vertex normals.
        for v in self.vertices:
            v.normal = np.zeros((3,1))
            for f in v.adjacent_faces():
                v.normal += f.normal * f.area
            v.normal /= norm(v.normal)

    def init_opengl(self):
        # Get per face vertex positions, normals, and colors.
        vertices = np.zeros((3, 3*len(self.faces)), dtype='float32')
        normals = np.zeros((3, 3*len(self.faces)), dtype='float32')
        colors = np.zeros((3, 3*len(self.faces)), dtype='float32')
        k = 0
        for i in range(len(self.faces)):
            f = self.faces[i]
            assert(f.index == i)
            h = f.halfedge
            assert(k == 3*f.index)
            while True:
                v = h.vertex
                vertices[:,k,None] = v.position
                normals[:,k,None]  = f.normal
                # normals[:,k,None]  = v.normal
                colors[:,k] = np.array([1.0, 0.5, 0.31], dtype='float32')
                # colors[:,k] = np.random.rand(3)
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
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def draw(self, shader):
        model = self.get_tf_world().matrix().T
        shader.set_mat4('model', model)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_elems_draw)
        # glDrawElements(GL_TRIANGLES, self.num_elems_draw, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_edges(self):
        glBegin(GL_LINES)
        glColor4f(1.0, 0.0, 0.0, 0.25)
        for e in self.edges:
            v0 = self.position(e.halfedge.vertex)
            v1 = self.position(e.halfedge.twin.vertex)
            glVertex3f(v0[0], v0[1], v0[2])
            glVertex3f(v1[0], v1[1], v1[2])
        glEnd()

    def draw_faces(self):
        white = (GLfloat * 4)(1., 1., 1., .5)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, white)
        glColor4f(1., 1., 1., 0.25)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        for f in self.faces:
            glBegin(GL_POLYGON)
            N = self.normal(f)
            glNormal3f(N[0], N[1], N[2])
            h = f.halfedge
            while True:
                v = self.position(h.vertex)
                glVertex3f(v[0], v[1], v[2])
                h = h.next
                if h is f.halfedge:
                    break
            glEnd()

    def build_convex(self, points):
        """Build convex halfedge mesh from input points.
        
        Arguments:
            points {np.ndarray} -- Input points (3xn).
        """
        num_points = points.shape[1]
        if num_points == 2:
            h0 = self.new_halfedge()
            h1 = self.new_halfedge()
            v0 = self.new_vertex()
            v1 = self.new_vertex()
            e = self.new_edge()
            h0.set_neighbors(h1, h1, v0, e, None)
            h1.set_neighbors(h0, h0, v1, e, None)
            v0.halfedge = h0
            v1.halfedge = h1
            e.halfedge = h0
        elif num_points >= 3:
            hull = ConvexHull(points.T)
            simplices = reorient(hull.simplices.tolist(), hull.neighbors.tolist(), list(hull.points))
            self.build(simplices, list(hull.points))
        self.reindex()

    def build(self, polygons, vertexPositions):
        # Maps a vertex index to the corresponding vertex.
        indexToVertex = dict()

        # Also store the vertex degree, i.e., the number of polygons that use
        # each vertex this information will be used to check that the mesh is
        # manifold.
        vertexDegree = dict()

        # Do some basic sanity checks on the input.
        for p in polygons:
            if len(p) < 3:
                raise RuntimeError('Error converting polygons to halfedge mesh:'
                                   'each polygon must have at least three vertices.')

            # We want to count the number of distinct vertex indices in this
            # polygon, to make sure it's the same as the number of vertices
            # in the polygon---if they disagree, then the polygon is not valid
            # (or at least, for simplicity we don't handle polygons of this type!).
            polygonIndices = set()

            # Loop over polygon vertices.
            for i in p:
                polygonIndices.add(i)
                # Allocate one vertex for each new index we encounter.
                if i not in indexToVertex.keys():
                    v = self.new_vertex()
                    indexToVertex[i] = v
                    vertexDegree[v] = 1 # We've now seen this vertex only once.
                else:
                    # Keep track of the number of times we've seen this vertex.
                    vertexDegree[indexToVertex[i]] += 1

            # Check that all the vertices of the current polygon are distinct.
            degree = len(p)
            if len(polygonIndices) < degree:
                raise RuntimeError('Error converting polygons to halfedge mesh: '
                    'one of the input polygons does not have distinct vertices {}', p)

        # The number of vertices in the mesh is the nubmer of unique indices
        # seen in the input.
        nVertices = len(indexToVertex)

        # The number of faces is the number of polygons in the input.
        nFaces = len(polygons)
        self.faces = [Face(self) for i in range(nFaces)]

        # We will store a map from ordered pairs of vertex indices to
        # the corresponding halfedge object in our new (halfedge) mesh
        # this map gets constructed during the next loop over polygons.
        pairToHalfedge = dict()

        # Next, we build the halfedge connectivity by again looping over
        # polygons.
        for p, f in zip(polygons, self.faces):

            faceHalfedges = [] # Cyclically ordered list of halfedges of this face.
            degree = len(p) # Number of vertices in this polgyon.

            # Loop over the halfedges of this face (ordered pairs of consecutive
            # vertices).
            for i in range(degree):
                a = p[i]
                b = p[(i+1)%degree]
                ab = (a,b)

                # Check if this halfedge already exists if so, we have a problem!
                if ab in pairToHalfedge.keys():
                    raise RuntimeError('Error converting polygons to halfedge mesh: found multiple '
                                       'oriented edges with indices {}.'.format(ab))
                else:
                    hab = self.new_halfedge()
                    pairToHalfedge[ab] = hab

                    # link the new halfedge to its face
                    hab.face = f
                    hab.face.halfedge = hab

                    # also link it to its starting vertex
                    hab.vertex = indexToVertex[a]
                    hab.vertex.halfedge = hab

                    # keep a list of halfedges in this face, so that we can
                    # later link them together in a loop (via their "next"
                    # pointers)
                    faceHalfedges.append(hab)

                # Also, check if the twin of this halfedge has already been
                # constructed (during construction of a different face).  If so,
                # link the twins together and allocate their shared halfedge.
                # By the end of this pass over polygons, the only halfedges that
                # will not have a twin will hence be those that sit along the
                # domain boundary.
                ba = (b,a)
                if ba in pairToHalfedge.keys():
                    hba = pairToHalfedge[ba]

                    # link the twins
                    hab.twin = hba
                    hba.twin = hab

                    # allocate and link their edge
                    e = self.new_edge()
                    hab.edge = e
                    hba.edge = e
                    e.halfedge = hab

            for i in range(degree):
                j = (i+1)%degree
                faceHalfedges[i].next = faceHalfedges[j]
        
        # For each vertex on the boundary, advance its halfedge pointer to one that
        # is also on the boundary.
        for v in self.vertices:
            h = v.halfedge
            while True:
                if h.twin is None:
                    v.halfedge = h
                    break
                h = h.twin.next
                if h is v.halfedge:
                    break
        
        # Construct faces for each boundary component.
        for h in self.halfedges:
            # Any halfedge that does not yet have a twin is on the boundary of
            # the domain. If we follow the boundary around long enough we will
            # of course eventually make a closed loop we can represent this
            # boundary loop by a new face. To make clear the distinction between
            # faces and boundary loops, the boundary face will (i) have a flag
            # indicating that it is a boundary loop, and (ii) be stored in a
            # list of boundaries, rather than the usual list of faces.  The
            # reason we need the both the flag *and* the separate list is that
            # faces are often accessed in two fundamentally different ways:
            # either by (i) local traversal of the neighborhood of some mesh
            # element using the halfedge structure, or (ii) global traversal of
            # all faces (or boundary loops).
            if h.twin is None:
                b = self.new_boundary()
                boundaryHalfedges = []
                # We now need to walk around the boundary, creating new
                # halfedges and edges along the boundary loop as we go.
                i = h
                while True:
                    # Create twin.
                    t = self.new_halfedge()
                    boundaryHalfedges.append(t)
                    i.twin = t
                    t.twin = i
                    t.face = b
                    t.vertex = i.next.vertex

                    # Create shared edge.
                    e = self.new_edge()
                    e.halfedge = i
                    i.edge = e
                    t.edge = e

                    # Advance i to the next halfedge along the current boundary
                    # loop by walking around its target vertex and stopping as
                    # soon as we find a halfedge that does not yet have a twin
                    # defined.
                    i = i.next
                    while i is not h and i.twin is not None:
                        i = i.twin.next
                    
                    if i is h:
                        break
                
                b.halfedge = boundaryHalfedges[0]

                # The only pointers that still need to be set are the "next"
                # pointers of the twins these we can set from the list of
                # boundary halfedges, but we must use the opposite order from
                # the order in the list, since the orientation of the boundary
                # loop is opposite the orientation of the halfedges "inside" the
                # domain boundary.
                degree = len(boundaryHalfedges)
                for p in range(degree):
                    q = (p-1+degree)%degree
                    boundaryHalfedges[p].next = boundaryHalfedges[q]
            
        # To make later traversal of the mesh easier, we will now advance the
        # halfedge associated with each vertex such that it refers to the
        # *first* non-boundary halfedge, rather than the last one.
        for v in self.vertices:
            v.halfedge = v.halfedge.twin.next

        # Finally, we check that all vertices are manifold.
        for v in self.vertices:
            # First check that this vertex is not a "floating" vertex
            # if it is then we do not have a valid 2-manifold surface.
            if v.halfedge is None:
                raise RuntimeError('Error converting polygon to halfedge mesh: '
                    'some vertices are not referenced by any polygon.')

            # Next, check that the number of halfedges emanating from this
            # vertex in our half edge data structure equals the number of
            # polygons containing this vertex, which we counted during our first
            # pass over the mesh.  If not, then our vertex is not a "fan" of
            # polygons, but instead has some other (nonmanifold) structure.
            count = 0
            h = v.halfedge
            while True:
                if not h.face.boundary:
                    count += 1
                h = h.twin.next
                if h is v.halfedge:
                    break

            if count != vertexDegree[v]:
                raise RuntimeError('Error converting polygon to halfedge mesh: '
                    'at least one vertex is non-manifold.')

        # Now that we have the connectivity, we copy the list of vertex
        # positions into member variables of the individual vertices.
        assert(len(indexToVertex.keys()) == len(self.vertices))
        for i in indexToVertex.keys():
            v = indexToVertex[i]
            v.position = vertexPositions[i].reshape((3,1))

        # Ensure triangle mesh.
        self.trianglulate()

        # Finally, index and cache mesh elements.
        self.init()

    def subdivide_4_1(self):
        for v in self.vertices:
            v.is_new = False
        for e in self.edges:
            e.is_new = False
        # Split all edges.
        n = len(self.edges)
        for i in range(n):
            e = self.edges[i]
            v = self.split_edge(e)
            v.is_new = True
            # v.position = e.new_position
            h0 = v.halfedge
            h = h0
            is_new = False
            while True:
                h.edge.is_new = is_new
                h = h.twin.next
                is_new = not is_new
                if h is h0:
                    break
        # Flip edges that connect an old and new vertex.
        for e in self.edges:
            v0 = e.halfedge.vertex
            v1 = e.halfedge.twin.vertex
            if e.is_new:
                if v0.is_new != v1.is_new:
                    self.flip_edge(e)

    def loop_subdivide(self):
        # Compute updated positions for all the vertices in the original mesh,
        # using the Loop subdivision rule.
        for v in self.vertices:
            s = np.zeros((3,1))
            h0 = v.halfedge
            h = h0
            n = 0
            while True:
                s += h.twin.vertex.position
                n += 1
                h = h.twin.next
                if h is h0:
                    break
            if n == 3:
                u = 3.0/16.0
            else:
                u = 3.0/(8.0*n)
            v.new_position = (1-n*u)*v.position + u*s
            v.is_new = False

        # Next, compute the updated vertex positions associated with edges.
        for e in self.edges:
            h0 = e.halfedge
            h1 = h0.next
            h2 = h1.next
            h3 = h0.twin
            h4 = h3.next
            h5 = h4.next
            v0 = h3.vertex
            v1 = h2.vertex
            v2 = h0.vertex
            v3 = h5.vertex
            e.new_position = 0.375*v0.position + 0.125*v1.position + 0.375*v2.position + 0.125*v3.position
            e.is_new = False

        # Next, we're going to split every edge in the mesh, in any order.  For
        # future reference, we're also going to store some information about
        # which subdivided edges come from splitting an edge in the original
        # mesh, and which edges are new. In this loop, we only want to iterate
        # over edges of the original mesh---otherwise, we'll end up splitting
        # edges that we just split (and the loop will never end!)
        n = len(self.edges)
        for i in range(n):
            e = self.edges[i]
            v = self.split_edge(e)
            v.is_new = True
            v.position = e.new_position
            h0 = v.halfedge
            h = h0
            is_new = False
            while True:
                h.edge.is_new = is_new
                h = h.twin.next
                is_new = not is_new
                if h is h0:
                    break

        # Finally, flip any new edge that connects an old and new vertex.
        for e in self.edges:
            v0 = e.halfedge.vertex
            v1 = e.halfedge.twin.vertex
            if e.is_new:
                if v0.is_new != v1.is_new:
                    self.flip_edge(e)

        # Copy the updated vertex positions to the subdivided mesh.
        for v in self.vertices:
            if not v.is_new:
                v.position = v.new_position

    def split_edge(self, e0):
        # This method splits the given edge and returns an iterator to the newly
        # inserted vertex. The halfedge of this vertex points along the edge
        # that was split, rather than the new edges.

        # Collect mesh elements.
        h1 = e0.halfedge
        h2 = h1.next
        h3 = h2.next
        h4 = h1.twin
        h5 = h4.next
        h6 = h5.next
        h7 = h6.twin
        h8 = h5.twin
        h9 = h3.twin
        h10 = h2.twin
        e1 = h2.edge
        e2 = h3.edge
        e3 = h5.edge
        e4 = h6.edge
        v1 = h4.vertex
        v2 = h3.vertex
        v3 = h5.vertex
        v4 = h6.vertex
        f1 = h2.face
        f2 = h4.face

        # Allocate new elements.
        h11 = self.new_halfedge()
        h12 = self.new_halfedge()
        h13 = self.new_halfedge()
        h14 = self.new_halfedge()
        h15 = self.new_halfedge()
        h16 = self.new_halfedge()
        e5 = self.new_edge()
        e6 = self.new_edge()
        e7 = self.new_edge()
        v0 = self.new_vertex()
        f3 = self.new_face()
        f4 = self.new_face()

        # Assign mesh elements.
        v0.halfedge = h11
        v0.position = (v1.position+v3.position)/2.0
        v1.halfedge = h12
        v3.halfedge = h1
        h1.next = h15
        h1.face = f1
        h15.set_neighbors(h3, h16, v0, e7, f1)
        h3.face = f1
        e7.halfedge = h15
        f1.halfedge = h15
        h5.next = h14
        h5.face = f2
        h14.set_neighbors(h4, h13, v4, e5, f2)
        h4.vertex = v0
        h4.face = f2
        e5.halfedge = h14
        f2.halfedge = h14
        h6.next = h12
        h6.face = f3
        h13.set_neighbors(h6, h14, v0, e5, f3)
        h12.set_neighbors(h13, h11, v1, e6, f3)
        e6.halfedge = h12
        f3.halfedge = h13
        h2.next = h16
        h2.face = f4
        h11.set_neighbors(h2, h12, v0, e6, f4)
        h16.set_neighbors(h11, h15, v2, e7, f4)
        f4.halfedge = h11

        return v0

    def flip_edge(self, e0):
        # This method flips the given edge and returns an iterator to the
        # flipped edge.

        # Get mesh elements.
        h0 = e0.halfedge
        h1 = h0.next
        h2 = h0.next
        while h2.next is not h0:
            h2 = h2.next
        h3 = h0.twin
        h4 = h3.next
        h5 = h4
        while h5.next is not h3:
            h5 = h5.next
        h6 = h1.twin
        h7 = h5.twin
        h8 = h4.twin
        h9 = h2.twin
        h10 = h1.next
        h11 = h4.next
        e1 = h5.edge
        e2 = h4.edge
        e3 = h2.edge
        e4 = h1.edge
        v0 = h3.vertex
        v1 = h6.vertex
        v2 = h2.vertex
        v3 = h0.vertex
        v4 = h8.vertex
        v5 = h5.vertex
        f0 = h0.face
        f1 = h3.face

        # Reassign mesh elements.
        h5.next = h1
        h1.next = h3
        h3.next = h11
        h2.next = h4
        h4.next = h0
        h0.next = h10
        h3.vertex = v1
        h0.vertex = v4
        h4.face = f0
        h1.face = f1
        f0.halfedge = h4
        f1.halfedge = h1

        return e0

    def trianglulate(self):
        for f in self.faces:
            self.split_polygon(f)

    def split_polygon(self, f):
        # Triangulate a polygonal face.

        # Collect mesh elements.
        h_int = []
        h0 = f.halfedge
        h = h0
        while True:
            h_int.append(h)
            h = h.next
            if h == h0:
                break
        n = len(h_int)
        if n == 3:
            return
        h1 = h0.next
        h2 = h1.next
        h3 = h_int[n-1]
        e0 = h0.edge
        e1 = h1.edge
        e2 = h2.edge
        e3 = h3.edge
        v0 = h0.vertex
        v1 = h1.vertex
        v2 = h2.vertex

        # Allocate mesh elements.
        f1 = self.new_face()
        h4 = self.new_halfedge()
        h5 = self.new_halfedge()
        e4 = self.new_edge()

        # Assign mesh elements.
        h0.face = f1
        h1.face = f1
        h1.next = h4
        h4.set_neighbors(h0, h5, v2, e4, f1)
        e4.halfedge = h4
        h5.set_neighbors(h2, h4, v0, e4, f)
        f1.halfedge = h0
        h3.next = h5
        f.halfedge = h3

        # Recurse.
        self.split_polygon(f)
        # showError("splitPolygon() not implemented.")

    def sample_points(self, n):
        # Compute total area.
        total_area = 0
        for f in self.faces:
            total_area += f.area
        points = np.zeros((3,n), dtype='float32')
        for i in range(n):
            # Sample face using area weights.
            r = total_area * np.random.rand()
            cum_sum = 0
            for f in self.faces:
                cum_sum += f.area
                if r <= cum_sum:
                    break
            # Sample uniform point on triangle.
            p0 = f.halfedge.vertex.position
            p1 = f.halfedge.next.vertex.position
            p2 = f.halfedge.next.next.vertex.position
            v1 = p1 - p0
            v2 = p2 - p0
            while True:
                a1 = np.random.rand()
                a2 = np.random.rand()
                if a1 + a2 <= 1:
                    break
            points[:,i,None] = p0 + a1*v1 + a2*v2
        return points
