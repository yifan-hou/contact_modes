import numpy as np
import contact_modes


def test_enum_contact_separate_3d():
    # Create four planar contact points.
    points = np.zeros((3,4))
    normals = np.zeros((3,4))
    points[:,0] = np.array([ 1, 1, 0])
    points[:,1] = np.array([-1, 1, 0])
    points[:,2] = np.array([-1,-1, 0])
    points[:,3] = np.array([ 1,-1, 0])
    normals[2,:] = 1.0
    print(points)
    print(normals)
    contact_modes.enumerate_contact_separating_3d(points, normals)
    assert(False)