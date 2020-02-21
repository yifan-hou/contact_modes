import numpy as np

from contact_modes._contact_modes import collide_2d
from contact_modes.viewer.backend import *

from .manifold import CollisionManifold


class CollisionManifold2D(CollisionManifold):
    def __init__(self):
        super(CollisionManifold2D, self).__init__()

    def draw(self, shader):
        shader.use()
        shader.set_mat4('model', np.eye(4))

        glPointSize(5)
        for i in range(len(self.dists)):
            glBegin(GL_POINTS)
            for p in self.pts_A.T:
                glVertex3f(p[0], p[1], 1)
            for p in self.pts_B.T:
                glVertex3f(p[0], p[1], 1)
            glEnd()

            glBegin(GL_LINES)
            for i in range(len(self.dists)):
                a = self.pts_A[:,i]
                b = self.pts_B[:,i]
                glVertex3f(a[0], a[1], 1)
                glVertex3f(b[0], b[1], 1)
            glEnd()

class CollisionManager2D(object):
    def __init__(self):
        self.pairs = []
        self.manifolds = []

    def add_pair(self, body_A, body_B):
        self.pairs.append((body_A, body_B))

    def get_manifolds(self):
        return self.manifolds
    
    def draw(self, shader):
        for m in self.manifolds:
            m.draw(shader)

    def collide(self):
        manifolds = []
        n_pairs = len(self.pairs)
        for i in range(n_pairs):
            body_A = self.pairs[i][0]
            body_B = self.pairs[i][1]
            body_A.reset_contacts()
            body_B.reset_contacts()
        for i in range(n_pairs):
            body_A = self.pairs[i][0]
            body_B = self.pairs[i][1]
            manifold = CollisionManifold2D()
            V_A = [v for v in body_A.get_collision_shape().vertices.T]
            V_B = [v for v in body_B.get_collision_shape().vertices.T]
            q_A = body_A.get_pose()
            q_B = body_B.get_pose()
            m = collide_2d(V_A, V_B, q_A, q_B)
            manifold.pts_A = np.array(m.pts_A).T
            manifold.pts_B = np.array(m.pts_B).T
            manifold.normal = np.array(m.normal)
            manifold.dists = np.array(m.dists)
            manifolds.append(manifold)
            # print('A', m.pts_A)
            # print('B', m.pts_B)
            # print('n', m.normal)
            # print('d', m.dists)
        self.manifolds = manifolds
        return manifolds
