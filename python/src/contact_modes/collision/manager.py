import numpy as np

from contact_modes import SE3, SO3
from contact_modes.shape import Sphere

from .gjk import gjk

DEBUG = True

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
            obstacle = p[1]
            manifold = gjk(sphere, obstacle)
            
            points[:,i,None] = pt_frame
            normals[:,i,None] = n_A
        
        return frame_centers, normals, tangents, dists
