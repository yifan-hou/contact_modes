import numpy as np
from numpy.linalg import norm

from contact_modes.collision import CollisionManager, TransformManager
from contact_modes.shape import Box, BoxWithHole, Cylinder, Torus

from .lattice import FaceLattice
from .util import get_color


def box_case(walls=1):
    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    target = Box()
    target.get_tf_world().set_translation(np.array([0, 0, 0.0]))
    target_wireframe = Box(1.0, 1.0, 1.0)
    target_wireframe.set_color(get_color('black'))
    target.set_wireframe(target_wireframe)

    x = np.array([1., 0, 0])
    y = np.array([0, 1., 0])
    z = np.array([0, 0, 1.])
    N = [z, y, x, -x, -y, -z]
    T = [(x,y), (z,x), (y,z), (z,y), (x,z), (y,x)]
    obstacles = []
    for i in range(walls):
        n = N[i]
        box_dims = np.array([10., 10., 10.])
        box_dims[np.where(np.abs(n) > 0)] = 1.0
        box = Box(*box_dims)
        box.get_tf_world().set_translation(-n)
        obstacles.append(box)

    # --------------------------------------------------------------------------
    # Collision manager
    # --------------------------------------------------------------------------
    manager = CollisionManager()
    for i in range(walls):
        for j in range(4):
            manager.add_pair(None, obstacles[i])
    
    # --------------------------------------------------------------------------
    # Contact points, normals, and tangents
    # --------------------------------------------------------------------------
    points = np.zeros((3, 4 * walls))
    normals = np.zeros((3, 4 * walls))
    tangents = np.zeros((3, 4 * walls, 2))
    k = 0
    for i in range(walls):
        for t_x in [1, -1]:
            for t_y in [1, -1]:
                points[:,k] = -0.5 * (N[i] + t_x * T[i][0] + t_y * T[i][1])
                normals[:,k] = N[i]
                tangents[:,k,0] = T[i][0]
                tangents[:,k,1] = T[i][1]
                k += 1
    
    return points, normals, tangents, target, obstacles, manager

def peg_in_hole(n=8):
    # --------------------------------------------------------------------------
    # Contact points, normals, and tangents
    # --------------------------------------------------------------------------
    radius = 0.30
    height = 2.0
    side_length = 1.0
    cylinder_scale = 1/1.5

    top_points = np.zeros((3, n))
    top_normals = np.zeros((3, n))
    top_tangents = np.zeros((3, n, 2))
    bot_points = np.zeros((3, n))
    bot_normals = np.zeros((3, n))
    bot_tangents = np.zeros((3, n, 2))
    theta = np.array([i/n*2*np.pi for i in range(n)])
    r = radius
    h = height * cylinder_scale
    l = side_length
    for i in range(n):
        t = theta[i]
        c = np.cos(t)
        s = np.sin(t)
        
        top_points[:,i] = np.array([r*c, r*s, h/2.0])
        bot_points[:,i] = np.array([r*c, r*s,-h/2.0])

        top_normals[:,i] = np.array([-c, -s, 0])
        bot_normals[:,i] = np.array([-c, -s, 0])

        top_tangents[:,i,0] = np.array([0.0, 0.0, 1.0])
        bot_tangents[:,i,0] = np.array([0.0, 0.0, 1.0])

        top_tangents[:,i,1] = np.cross(top_normals[:,i], top_tangents[:,i,0])
        bot_tangents[:,i,1] = np.cross(bot_normals[:,i], bot_tangents[:,i,0])

    points = np.concatenate((top_points, bot_points), axis=1)
    normals = np.concatenate((top_normals, bot_normals), axis=1)
    tangents = np.concatenate((top_tangents, bot_tangents), axis=1)

    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    target = Cylinder(radius, height * cylinder_scale)
    hole = BoxWithHole(radius, side_length, height)

    return points, normals, tangents, target, [hole]

class TorusPuzzleManager(object):
    def __init__(self, n):
        self.n = n

    def closest_points(self, points, normals, tangents, tf):
        # Transform object points.
        points = SE3.transform_point(tf, points)
        # Transform object normals.
        normals = SO3.transform_point(tf.R, normals)
        # Transform object tangents.
        n_pts = points.shape[1]
        for i in range(n_pts):
            tangents[:,i,0] = SO3.transform_point(tf.R, tangents[:,i,0]).flatten()
            tangents[:,i,1] = SO3.transform_point(tf.R, tangents[:,i,1]).flatten()
        # Get distances.
        dists = np.zeros((n_pts,))

        return points, normals, tangents, dists

def torus_puzzle(n=8):
    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    target = Torus(strip_color=get_color('clay'))
    target.get_tf_world().set_translation(np.array([0, 0, 0.0]))

    obstacle = Torus(strip_color=get_color('teal'))

    # --------------------------------------------------------------------------
    # Collision manager
    # --------------------------------------------------------------------------
    manager = TorusPuzzleManager(n)
    
    # --------------------------------------------------------------------------
    # Contact points, normals, and tangents
    # --------------------------------------------------------------------------
    points = np.zeros((3, n))
    normals = np.zeros((3, n))
    tangents = np.zeros((3, n, 2))
    
    return points, normals, tangents, target, obstacles, manager