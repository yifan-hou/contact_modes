import numpy as np

from contact_modes._contact_modes import collide_2d


from .manifold import CollisionManifold

class CollisionManifold2D(CollisionManifold):
    def __init__(self):
        super(CollisionManifold2D, self).__init__()

class CollisionManager2D(object):
    def __init__(self):
        self.pairs = []
        self.manifolds = []

    def add_pair(self, body_A, body_B):
        self.pairs.append((body_A, body_B))

    def get_manifolds(self):
        return self.manifolds

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
            V_A = [v for v in body_A.get_shape().vertices.T]
            V_B = [v for v in body_B.get_shape().vertices.T]
            q_A = body_A.get_pose()
            q_B = body_B.get_pose()
            m = collide_2d(V_A, V_B, q_A, q_B)
            
