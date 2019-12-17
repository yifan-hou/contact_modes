import numpy as np
import contact_modes
from contact_modes import enumerate_contact_separating_3d_exponential


def test_enum_contact_separate_3d():
    # Create four planar contact points.
    points = np.zeros((3,4))
    normals = np.zeros((3,4))
    points[:,0] = np.array([ 1, 1, 0])
    points[:,1] = np.array([-1, 1, 0])
    points[:,2] = np.array([-1,-1, 0])
    points[:,3] = np.array([ 1,-1, 0])
    normals[2,:] = 1.0
    # print(points)
    # print(normals)
    # contact_modes.enumerate_contact_separating_3d(points, normals)
    modes = enumerate_contact_separating_3d_exponential(points, normals)
    print(modes)

    # Create contact manifold in the shape of an octagon.
    n = 8
    points = np.zeros((3,n))
    normals = np.zeros((3,n))
    for i in range(n):
        points[0,i] = np.cos(i/8*2*np.pi)
        points[1,i] = np.sin(i/8*2*np.pi)
        normals[2,i] = 1.0
    modes = enumerate_contact_separating_3d_exponential(points, normals)
    print(modes)

    # Create 

    assert(False)