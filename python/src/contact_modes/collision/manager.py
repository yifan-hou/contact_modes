import numpy as np

from contact_modes import SE3, SO3
from contact_modes.shape import Icosphere

from .gjk import gjk

DEBUG = False

class CollisionManager(object):
    def __init__(self):
        self.pairs = []

    def add_pair(self, point, obs):
        # Create contact point shape.
        sphere = Icosphere(radius=0.1, refine=0)
        sphere.get_tf_world().set_translation(point)
        # Add pair.
        self.pairs.append((sphere, obs))

    def closest_points(self, points, normals, tangents, tf):
        if DEBUG:
            print(tf)

        # Transform object points.
        points = SE3.transform_point(tf, points)
        if DEBUG:
            print('points')
            print(points)

        # Compute closest points.
        n_pts = points.shape[1]
        n_pairs = len(self.pairs)
        dists = np.zeros((n_pts,))
        frame_centers = np.zeros((3, n_pts))
        for i in range(n_pairs):
            p = self.pairs[i]
            sphere   = p[0]
            sphere.get_tf_world().set_translation(points[:,i,None])
            obstacle = p[1]
            manifold = gjk(obstacle, sphere)

            if DEBUG:
                print(manifold.pts_A)
                print(manifold.pts_B)
                print(manifold.normal)
                print(manifold.dist)
            
            points[:,i,None]  = manifold.pts_A
            normals[:,i,None] = manifold.normal
            dists[i] = manifold.dist + sphere.margin()
        
        return points, normals, tangents, dists