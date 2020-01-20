from time import time

import numpy as np

import contact_modes
from contact_modes import (FaceLattice, enumerate_contact_separating_3d,
                           enumerate_contact_separating_3d_exponential)


np.set_printoptions(precision=8, suppress=None)

def gen_box_ground():
    # Create four planar contact points.
    points = np.zeros((3,4))
    normals = np.zeros((3,4))
    points[:,0] = np.array([ 1, 1, 0])
    points[:,1] = np.array([-1, 1, 0])
    points[:,2] = np.array([-1,-1, 0])
    points[:,3] = np.array([ 1,-1, 0])
    normals[2,:] = 1.0

    return points, normals

def gen_octagon_ground():
    n = 6
    points = np.zeros((3,n))
    normals = np.zeros((3,n))
    for i in range(n):
        points[0,i] = np.cos(i/8*2*np.pi)
        points[1,i] = np.sin(i/8*2*np.pi)
        normals[2,i] = 1.0
    return points, normals

def gen_box_wall():
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
    return points, normals

def gen_box_sandwich():
    pass

def test_sample_twist():
    np.set_printoptions(precision=8, suppress=True)
    points, normals = gen_box_ground()
    A, b = contact_modes.contacts_to_half(points, normals)
    modes = enumerate_contact_separating_3d_exponential(points, normals)
    print(modes)
    for m in modes:
        xi = contact_modes.sample_twist_contact_separating(points, normals, m)
        print(xi)
    assert(False)


def test_enum_contact_separate_3d():
    np.set_printoptions(precision=8, suppress=None)

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
    #
    modes = contact_modes.enumerate_contact_separating_3d_exponential(points, normals)
    print('contact mode: ')
    print(modes)

    # modes = enumerate_contact_separating_3d(points, normals)

    modes, lattice = enumerate_contact_separating_3d(points, normals)
    print(modes)

    #assert(False)
    #return

    # Create contact manifold in the shape of an octagon.
    n = 6
    points = np.zeros((3,n))
    normals = np.zeros((3,n))
    for i in range(n):
        points[0,i] = np.cos(i/8*2*np.pi)
        points[1,i] = np.sin(i/8*2*np.pi)
        normals[2,i] = 1.0
    modes = contact_modes.enumerate_contact_separating_3d(points, normals)
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
    modes = contact_modes.enumerate_contact_separating_3d(points, normals)
    print('time', time() - t_start)

    print(modes)
    #print(modes.shape)

    t_start = time()
    # modes = enumerate_contact_separating_3d(points, normals)
    enumerate_contact_separating_3d(points, normals)
    print('time 3d', time() - t_start)

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
    modes = contact_modes.enumerate_contact_separating_3d(points, normals)
    print(modes)
    modes = enumerate_contact_separating_3d(points, normals)
    assert(False)
    

def test_enum_contact_all_3d():

    # Create four planar contact points.
    points = np.zeros((3, 4))
    normals = np.zeros((3, 4))
    points[:, 0] = np.array([1, 1, 0])
    points[:, 1] = np.array([-1, 1, 0])
    points[:, 2] = np.array([-1, -1, 0])
    points[:, 3] = np.array([1, -1, 0])
    normals[2, :] = 1.0
    # print(points)
    # print(normals)
    #
    tangentials = np.zeros((3,4,2))
    tangentials[0,:,0] = 1
    tangentials[1,:,1] = 1
    modes_1 = contact_modes.enumerate_contact_separating_3d_exponential(points, normals)
    #modes_2 = contact_modes.enumerate_all_modes_3d_exponential(points, normals,tangentials,2)
    modes,lattice = contact_modes.enumerate_all_modes_3d(points, normals,tangentials,4)
    print('contact mode: ')
    print(modes)
    print(len(modes))
    '''
    # Create contact manifold in the shape of an octagon.
    n = 6
    points = np.zeros((3,n))
    normals = np.zeros((3,n))
    for i in range(n):
        points[0,i] = np.cos(i/8*2*np.pi)
        points[1,i] = np.sin(i/8*2*np.pi)
        normals[2,i] = 1.0
    tangentials = np.zeros((3,n,2))
    tangentials[0,:,0] = 1
    tangentials[1,:,1] = 1
    modes = contact_modes.enumerate_all_modes_3d(points, normals, tangentials, 4)
    print('contact mode: ')
    print(modes)
    print(len(modes))
    
    # Create box-against-wall contact manifold.
    points = np.zeros((3, 8))
    normals = np.zeros((3, 8))
    # box on x-y plane
    points[:, 0] = np.array([1, 1, 0])
    points[:, 1] = np.array([-1, 1, 0])
    points[:, 2] = np.array([-1, -1, 0])
    points[:, 3] = np.array([1, -1, 0])
    # box against x-z wall
    points[:, 4] = np.array([1, 1, 2])
    points[:, 5] = np.array([-1, 1, 2])
    points[:, 6] = np.array([-1, 1, 0])
    points[:, 7] = np.array([1, 1, 0])

    normals[2, 0:4] = 1.0
    normals[1, 4:8] = -1.0

    tangentials = np.zeros((3, 8, 2))
    tangentials[0, 0:4, 0] = 1
    tangentials[1, 0:4, 1] = 1
    tangentials[0, 4:8, 0] = 1
    tangentials[2, 4:8, 1] = -1

    t_start = time()
    modes = contact_modes.enumerate_all_modes_3d_exponential(points, normals,tangentials,4)
    print('time', time() - t_start)
    print(modes)
    print(len(modes))
    # print(modes.shape)
    '''

test_enum_contact_all_3d()