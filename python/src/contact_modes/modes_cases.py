import numpy as np
from numpy.linalg import norm

from contact_modes.collision import CollisionManager
from contact_modes.shape import Box, BoxWithHole, Cylinder

from .polytope import FaceLattice
from .util import get_color

# from contact_modes import (CollisionManager, FaceLattice,
#                            enumerate_contact_separating_3d, get_color,
#                            get_data, sample_twist_contact_separating)
# from contact_modes.viewer import (SE3, Application, Arrow, Box, BoxWithHole,
#                                   Cylinder, OITRenderer, Shader, Viewer,
#                                   Window)


def box_ground():
    # --------------------------------------------------------------------------
    # Contact points, normals, and tangents
    # --------------------------------------------------------------------------
    points = np.zeros((3,4))
    normals = np.zeros((3,4))
    points[:,0] = np.array([ 0.5, 0.5, -0.5])
    points[:,1] = np.array([-0.5, 0.5, -0.5])
    points[:,2] = np.array([-0.5,-0.5, -0.5])
    points[:,3] = np.array([ 0.5,-0.5, -0.5])
    normals[2,:] = 1.0
    tangents = np.zeros((3, 4, 2))
    tangents[0, :, 0] = 1
    tangents[1, :, 1] = 1

    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    target = Box()
    target.get_tf_world().set_translation(np.array([0, 0, 0.5]))
    target_wireframe = Box(1.0 + 1e-4, 1.0 + 1e-4, 1.0 + 1e-4)
    target_wireframe.set_color(get_color('black'))
    target.set_wireframe(target_wireframe)

    ground = Box(10, 10, 1.0)
    ground.get_tf_world().set_translation(np.array([0, 0, -0.5]))

    # --------------------------------------------------------------------------
    # Collision manager
    # --------------------------------------------------------------------------
    manager = CollisionManager()
    for i in range(4):
        manager.add_pair(points[:,i], ground)

    return points, normals, tangents, target, [ground], manager

def box_wall():
    # --------------------------------------------------------------------------
    # Contact points and normals
    # --------------------------------------------------------------------------
    points = np.zeros((3,8))
    normals = np.zeros((3,8))
    tangents = np.zeros((3, 8, 2))
    # box on x-y plane
    points[:,0] = np.array([ 0.5, 0.5, -0.5])
    points[:,1] = np.array([-0.5, 0.5, -0.5])
    points[:,2] = np.array([-0.5,-0.5, -0.5])
    points[:,3] = np.array([ 0.5,-0.5, -0.5])
    normals[2,0:4] = 1.0
    tangents[0, 0:4, 0] = 1
    tangents[1, 0:4, 1] = 1
    # box against x-z wall
    points[:,4] = np.array([ 0.5, 0.5,  0.5])
    points[:,5] = np.array([-0.5, 0.5,  0.5])
    points[:,6] = np.array([-0.5, 0.5, -0.5])
    points[:,7] = np.array([ 0.5, 0.5, -0.5])
    normals[1,4:8] =-1.0
    tangents[0, 4:8, 0] = 1
    tangents[2, 4:8, 1] = 1

    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    target = Box()
    target.get_tf_world().set_translation(np.array([0, 0, 0.5]))
    target_wireframe = Box(1.0, 1.0, 1.0)
    target_wireframe.set_color(get_color('black'))
    target.set_wireframe(target_wireframe)

    ground = Box(10, 10, 1.0)
    ground.get_tf_world().set_translation(np.array([0, 0, -0.5]))
    wall = Box(10, 1, 10)
    wall.get_tf_world().set_translation(np.array([0, 1.0, 5]))

    # --------------------------------------------------------------------------
    # Collision manager
    # --------------------------------------------------------------------------
    manager = CollisionManager()
    for i in range(4):
        manager.add_pair(points[:,i], ground)
    for i in range(4, 8):
        manager.add_pair(points[:,i], wall)

    return points, normals, tangents, target, [ground, wall], manager

def box_corner():
    pass

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
    target = Cylinder(radius - 1e-4, height * cylinder_scale)
    hole = BoxWithHole(radius, side_length, height)

    return points, normals, tangents, target, [hole]
