from time import time

import numpy as np

import contact_modes
from contact_modes import (FaceLattice, enumerate_contact_separating_3d,
                           enumerate_contact_separating_3d_exponential)


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
    # modes = enumerate_contact_separating_3d(points, normals)

    # Create contact manifold in the shape of an octagon.
    n = 6
    points = np.zeros((3,n))
    normals = np.zeros((3,n))
    for i in range(n):
        points[0,i] = np.cos(i/8*2*np.pi)
        points[1,i] = np.sin(i/8*2*np.pi)
        normals[2,i] = 1.0
    modes = enumerate_contact_separating_3d_exponential(points, normals)
    print(modes)
    # modes = enumerate_contact_separating_3d(points, normals)

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

    t_start = time()
    modes = enumerate_contact_separating_3d_exponential(points, normals)
    print('time', time() - t_start)
    print(modes)
    print(modes.shape)
    t_start = time()
    # modes = enumerate_contact_separating_3d(points, normals)
    enumerate_contact_separating_3d(points, normals)
    print('time', time() - t_start)

    # Box against wall - Polar.
    M = np.array([[1, 1, 0, 1, 1, 0],
                  [1, 1, 0, 1, 0, 1],
                  [0, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 0],
                  [1, 0, 0, 1, 1, 1],
                  [1, 1, 0, 0, 1, 1],
                  [0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 1, 1, 1]])
    d = 4
    
    t_start = time()
    L = FaceLattice(M, d)
    print('time', time() - t_start)
    
    print(L.num_proper_faces())
    print(L.num_faces())
    # print(L.mode_strings())
    # print(modes)
    modes1 = L.mode_strings()
    # for i in range(modes.shape[0]):
    #     print(modes[i,:])
    #     print(modes1[i,:])
    #     print(modes[i,:] == modes1[i,:])
    m0 = set([tuple(m) for m in modes])
    m1 = set([tuple(m) for m in modes1])
    print(m0.difference(m1))
    # print(modes.tolist())

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
    modes = enumerate_contact_separating_3d_exponential(points, normals)
    print(modes)
    # modes = enumerate_contact_separating_3d(points, normals)

    assert(False)
