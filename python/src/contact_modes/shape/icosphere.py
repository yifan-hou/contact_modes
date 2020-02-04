import numpy as np

from .halfedgemesh import HalfedgeMesh

from contact_modes.viewer.backend import *

class Icosphere(HalfedgeMesh):
    def __init__(self, radius=1.0, refine=2):
        super(Icosphere, self).__init__()

        self.radius = radius

        vertexPositions = []
        t = (1.0+np.sqrt(5.0))/2.0

        vertexPositions.append(np.array([-1, t, 0]).reshape((3,1)))
        vertexPositions.append(np.array([ 1, t, 0]).reshape((3,1)))
        vertexPositions.append(np.array([-1,-t, 0]).reshape((3,1)))
        vertexPositions.append(np.array([ 1,-t, 0]).reshape((3,1)))

        vertexPositions.append(np.array([ 0,-1, t]).reshape((3,1)))
        vertexPositions.append(np.array([ 0, 1, t]).reshape((3,1)))
        vertexPositions.append(np.array([ 0,-1,-t]).reshape((3,1)))
        vertexPositions.append(np.array([ 0, 1,-t]).reshape((3,1)))

        vertexPositions.append(np.array([ t, 0,-1]).reshape((3,1)))
        vertexPositions.append(np.array([ t, 0, 1]).reshape((3,1)))
        vertexPositions.append(np.array([-t, 0,-1]).reshape((3,1)))
        vertexPositions.append(np.array([-t, 0, 1]).reshape((3,1)))

        for i in range(len(vertexPositions)):
            vertexPositions[i] *= 1.0/np.linalg.norm(vertexPositions[i])

        polygons = []

        # 5 faces around point 0
        polygons.append([0, 11, 5])
        polygons.append([0, 5, 1])
        polygons.append([0, 1, 7])
        polygons.append([0, 7, 10])
        polygons.append([0, 10, 11])

        # 5 adjacent faces
        polygons.append([1, 5, 9])
        polygons.append([5, 11, 4])
        polygons.append([11, 10, 2])
        polygons.append([10, 7, 6])
        polygons.append([7, 1, 8])

        # 5 faces around point 3
        polygons.append([3, 9, 4])
        polygons.append([3, 4, 2])
        polygons.append([3, 2, 6])
        polygons.append([3, 6, 8])
        polygons.append([3, 8, 9])

        # 5 adjacent faces
        polygons.append([4, 9, 5])
        polygons.append([2, 4, 11])
        polygons.append([6, 2, 10])
        polygons.append([8, 6, 7])
        polygons.append([9, 8, 1])

        self.build(polygons, vertexPositions)

        # refine
        for i in range(refine):
            self.refine()

        self.init()

    def refine(self):
        self.loop_subdivide()

        for v in self.vertices:
            v.position *= 1.0/np.linalg.norm(v.position)

    def set_radius(self, radius):
        self.radius = radius

    def draw(self, shader):
        shader.use()

        # Compute scaled transforms for the arrow.
        g = self.get_tf_world().matrix()
        s = np.diag([self.radius, self.radius, self.radius, 1.0])
        shader.set_mat4('model', (g @ s).T)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_elems_draw)
        glBindVertexArray(0)

    def supmap(self, v):
        return self.get_tf_world().t

    def margin(self):
        return self.radius