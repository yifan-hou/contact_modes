import numpy as np
from numpy.linalg import norm

from contact_modes import (FaceLattice, enumerate_contact_separating_3d,
                           get_color, get_data,
                           sample_twist_contact_separating)
from contact_modes.viewer import (SE3, Application, Box, OITRenderer, Shader,
                                  Viewer, Window, Arrow, Cylinder)


def box_ground():
    # --------------------------------------------------------------------------
    # Contact points, normals, and tangents
    # --------------------------------------------------------------------------
    points = np.zeros((3,4))
    normals = np.zeros((3,4))
    points[:,0] = np.array([ 0.5, 0.5, 0])
    points[:,1] = np.array([-0.5, 0.5, 0])
    points[:,2] = np.array([-0.5,-0.5, 0])
    points[:,3] = np.array([ 0.5,-0.5, 0])
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

    return points, normals, tangents, target, [ground]

def box_wall():
    # --------------------------------------------------------------------------
    # Contact points and normals
    # --------------------------------------------------------------------------
    points = np.zeros((3,8))
    normals = np.zeros((3,8))
    tangents = np.zeros((3, 8, 2))
    # box on x-y plane
    points[:,0] = np.array([ 0.5, 0.5, 0])
    points[:,1] = np.array([-0.5, 0.5, 0])
    points[:,2] = np.array([-0.5,-0.5, 0])
    points[:,3] = np.array([ 0.5,-0.5, 0])
    normals[2,0:4] = 1.0
    tangents[0, 0:4, 0] = 1
    tangents[1, 0:4, 1] = 1
    # box against x-z wall
    points[:,4] = np.array([ 0.5, 0.5, 1])
    points[:,5] = np.array([-0.5, 0.5, 1])
    points[:,6] = np.array([-0.5, 0.5, 0])
    points[:,7] = np.array([ 0.5, 0.5, 0])
    normals[1,4:8] =-1.0
    tangents[0, 4:8, 0] = 1
    tangents[2, 4:8, 1] = 1

    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    target = Box()
    target.get_tf_world().set_translation(np.array([0, 0, 0.5+1e-4]))
    target_wireframe = Box(1.0 + 1e-4, 1.0 + 1e-4, 1.0 + 1e-4)
    target_wireframe.set_color(get_color('black'))
    target.set_wireframe(target_wireframe)

    ground = Box(10, 10, 1.0)
    ground.get_tf_world().set_translation(np.array([0, 0, -0.5]))
    wall = Box(10, 1, 10)
    wall.get_tf_world().set_translation(np.array([0, 1.0+1e-4, 5+1e-4]))

    return points, normals, tangents, target, [ground, wall]

def box_corner():
    pass