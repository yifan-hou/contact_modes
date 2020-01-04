import numpy as np
import contact_modes as cm


def test_affine_dim():
    # Create four planar contact points.
    points = np.zeros((3,4))
    normals = np.zeros((3,4))
    points[:,0] = np.array([ 1, 1, 0])
    points[:,1] = np.array([-1, 1, 0])
    points[:,2] = np.array([-1,-1, 0])
    points[:,3] = np.array([ 1,-1, 0])
    normals[2,:] = 1.0
    print(cm.affine_dim(cm.contacts_to_half(points, normals)[0]))

    # Create contact manifold in the shape of an octagon.
    n = 6
    points = np.zeros((3,n))
    normals = np.zeros((3,n))
    for i in range(n):
        points[0,i] = np.cos(i/8*2*np.pi)
        points[1,i] = np.sin(i/8*2*np.pi)
        normals[2,i] = 1.0

    # Create box-against-wall contact manifold.
    points = np.zeros((3,8))
    normals = np.zeros((3,8))
    # box on x-y plane
    points[:,0] = np.array([ 1, 1, 0])
    points[:,1] = np.array([-1, 1, 0])
    points[:,2] = np.array([-1,-1, 0])
    points[:,3] = np.array([ 1,-1, 0])
    # box against x-z wall
    points[:,4] = np.array([ 1, 1, 2])
    points[:,5] = np.array([-1, 1, 2])
    points[:,6] = np.array([-1, 1, 0])
    points[:,7] = np.array([ 1, 1, 0])
    normals[2,0:4] = 1.0
    normals[1,4:8] =-1.0
    print(cm.affine_dim(cm.contacts_to_half(points, normals)[0]))

    # Create box-sandwich contact manifold.
    points = np.zeros((3,8))
    normals = np.zeros((3,8))
    points[:,0] = np.array([ 1, 1, 0])
    points[:,1] = np.array([-1, 1, 0])
    points[:,2] = np.array([-1,-1, 0])
    points[:,3] = np.array([ 1,-1, 0])
    points[:,4] = np.array([ 1, 1, 1])
    points[:,5] = np.array([-1, 1, 1])
    points[:,6] = np.array([-1,-1, 1])
    points[:,7] = np.array([ 1,-1, 1])
    normals[2,0:4] = 1.0
    normals[2,4:8] =-1.0
    print(cm.affine_dim(cm.contacts_to_half(points, normals)[0]))

    assert(False)

def test_linear_dim():
    pass