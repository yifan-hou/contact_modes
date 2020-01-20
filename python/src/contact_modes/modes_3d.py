
# -*- coding: utf-8 -*-

from time import time

import numpy as np
import pyhull
import scipy as sp
from pyhull.halfspace import Halfspace
from .helpers import hat, lexographic_combinations, exp_comb, zenotope_vertex, feasible_faces
from .interior_point import int_pt_cone, interior_point_halfspace
from .polytope import FaceLattice

DEBUG = False


def contacts_to_half(points, normals):
    # Create halfspace inequalities, Ax - b ≥ 0.
    n_pts = points.shape[1]
    A = np.zeros((n_pts, 6))
    for i in range(n_pts):
        A[i,0:3] = normals[:,i].flatten()
        A[i,3:6] = np.dot(normals[:,i].T, hat(points[:,i])).flatten()
    A *= -1
    b = np.zeros((n_pts, 1))
    return A, b

def sample_twist_contact_separating(points, normals, modestr):
    A, b = contacts_to_half(points, normals)

    # print('mode', modestr)
    c = np.where(modestr == 'c')[0]
    # print(c)

    n_pts = points.shape[1]

    mask = np.zeros(n_pts, dtype=bool)
    mask[c] = 1
    if DEBUG:
        print('mask', mask)
    C = A[mask, :]
    H = A[~mask, :]

    # Project into null space.
    if np.sum(mask) > 0:
        null = sp.linalg.null_space(C)
        H = np.dot(H, null)
        # print(H.shape)
        x = null @ int_pt_cone(H)
        # return null @ int_pt_cone(H)
        print(A @ x)
        return x
    else:
        return int_pt_cone(H)

def sample_twist_sliding_sticking(points, normals, modestr):
    pass

def enumerate_contact_separating_3d_exponential(points, normals):
    # Check inputs dimensions.
    assert(points.shape[1] == normals.shape[1])
    assert(points.shape[0] == 3)
    assert(normals.shape[0] == 3)

    # Get dimensions.2
    n_pts = points.shape[1]

    # Create halfspace inequalities, Ax - b ≥ 0.
    n_pts = points.shape[1]
    A = np.zeros((n_pts, 6))
    for i in range(n_pts):
        A[i,0:3] = normals[:,i].flatten()
        A[i,3:6] = np.dot(normals[:,i].T, hat(points[:,i])).flatten()
    # A *= -1
    b = np.zeros((n_pts, 1))
    if DEBUG:
        print('A')
        print(A)

    # Enumerate contact modes and check for feasibility.
    modes = []
    for i in range(0, n_pts+1):
        for c in lexographic_combinations(n_pts, i):
            if DEBUG:
                print('c', c)
            mask = np.zeros(n_pts, dtype=bool)
            mask[c] = 1
            if DEBUG:
                print('mask', mask)
            C = A[mask, :]
            H = A[~mask, :]

            # Skip case where all contacts active.
            if np.sum(mask) == n_pts:
                m = np.array(['s'] * n_pts)
                m[c] = 'c'
                modes.append(m.tolist())
                if DEBUG:
                    print('Appending mode', m.tolist())
                continue

            # Project into null space.
            if np.sum(mask) > 0:
                null = sp.linalg.null_space(C)
                H = np.dot(H, null)
                if DEBUG:
                    print('projecting H into null space')
                    print(H)

            # Compute interior point.
            # b = np.zeros((H.shape[0], 1))
            # x = interior_point_halfspace(H, b)
            # print('halfspace')
            # print(c)
            # print(x)
            # print(H @ x)
            x = int_pt_cone(H)
            # print('cone')
            # print(x)
            # print(H @ x)

            # If point is strictly interior, then the mode string is valid.
            if 1e-5 < np.linalg.norm(np.dot(H, x)):
                m = np.array(['s'] * n_pts)
                m[c] = 'c'
                modes.append(m.tolist())
                if DEBUG:
                    print('Appending mode', m.tolist())

    # return np.array(sorted(modes))
    return np.array(modes)


def enumerate_contact_separating_3d(points, normals):
    # Check inputs dimensions.
    assert(points.shape[1] == normals.shape[1])
    assert(points.shape[0] == 3)
    assert(normals.shape[0] == 3)

    n_pts = points.shape[1]

    # Create halfspace inequalities, Ax - b <= 0.
    A, b = contacts_to_half(points, normals)
    if DEBUG:
        print('A')
        print(A)

    # Get interior point using linear programming.
    # int_pt = interior_point_halfspace(A, b) # Wow such math
    # print('int_pt')
    # print(int_pt, '\n' , A @ int_pt)

    # Get interior point using SVD.
    int_pt = int_pt_cone(A)
    if DEBUG:
        print('int_pt2')
        print(int_pt, '\n', A @ int_pt)

    # Compute dual points.
    b_off = b - np.dot(A, int_pt)
    dual = A / b_off
    if DEBUG:
        print('b off')
        print(b_off)
        print('dual')
        print(dual)

    # Project dual points into affine space.
    null = sp.linalg.null_space((dual - dual[1,:]))
    orth = sp.linalg.orth((dual - dual[1,:]).T)
    if orth.shape[1] != 6:
        dual = np.dot((dual-dual[1,:]), orth)
        if DEBUG:
            print('orth @ dual')
            print(dual)
    if DEBUG:
        print('null')
        print(null)
        print('orth')
        print(orth)

    # Compute dual convex hull.
    dual = [list(dual[i,:]) for i in range(n_pts)]
    ret = pyhull.qconvex('Fv', dual)
    if DEBUG:
        print('dual')
        print(np.array(dual))
        print(np.array(ret))

    # Build facet-vertex incidence matrix.
    n_facets = int(ret[0])
    M = np.zeros((n_pts, n_facets), int)
    for i in range(1, len(ret)):
        vert_set = [int(x) for x in ret[i].split(' ')][1:]
        for v in vert_set:
            M[v,i-1] = 1
    if DEBUG:
        print('M')
        print(M)

    # Build face lattice.
    L = FaceLattice(M, len(dual[0]))

    # Return mode strings.
    return L.mode_strings(), L

def enum_sliding_sticking_3d(points, normals, cs_mode):
    pass


def enumerate_all_modes_3d_exponential(points, normals, tangentials, num_sliding_plane):
    # Check inputs dimensions.
    assert (points.shape[1] == normals.shape[1])
    assert (points.shape[0] == 3)
    assert (normals.shape[0] == 3)
    assert (tangentials.shape[0] == 3)
    assert (tangentials.shape[2] == 2) #  two tangentials vectors: x and y

    # Get dimensions.2
    n_pts = points.shape[1]

    # Create halfspace inequalities, Ax - b ≥ 0.
    n_pts = points.shape[1]
    A = np.zeros((n_pts, 6))
    for i in range(n_pts):
        A[i, 0:3] = normals[:, i].flatten()
        A[i, 3:6] = np.dot(normals[:, i].T, hat(points[:, i])).flatten()
    # A *= -1
    b = np.zeros((n_pts, 1))
    if DEBUG:
        print('A')
        print(A)
    A = -A

    # Get linearized sliding sections from number of sliding modes
    #assert (num_sliding_modes % 2 == 0)
    D = np.array([[np.cos(2*np.pi*i/(2*num_sliding_plane)),np.sin(2*np.pi*i/(2*num_sliding_plane)),0] for i in range(2*num_sliding_plane)])
    # Get sliding mode section enumeration
    sliding_sections = np.identity(2*num_sliding_plane,dtype = int) + np.roll(np.identity(2*num_sliding_plane,dtype = int),num_sliding_plane-1,axis=1)
    sliding_lines = np.identity(2*num_sliding_plane,dtype = int) + np.roll(np.identity(2*num_sliding_plane,dtype = int),num_sliding_plane,axis=1)

    T = np.zeros((n_pts,num_sliding_plane*2,6))
    T_section = np.zeros((n_pts,num_sliding_plane*2,2,6))
    T_line = np.zeros((n_pts,num_sliding_plane*2,2,6))# sliding plane normals
    for i in range(n_pts):
        R = np.concatenate((tangentials[:, i, :],normals[:, i].reshape(-1,1)), axis=1)
        for j in range(num_sliding_plane*2):
            T_i = np.dot(R,D[j])
            T[i,j,0:3] = T_i
            T[i,j,3:6] = np.dot(T_i, hat(points[:, i]))
    for i in range(n_pts):
        for j in range(num_sliding_plane*2):
            T_section[i,j,:,:] = T[i,sliding_sections[j]==1]
            T_line[i,j,:,:] = T[i,sliding_lines[j]==1]
    T_sliding = np.concatenate((T_section,T_line),axis=1)
    num_sliding_modes = 4*num_sliding_plane


    # Enumerate contact modes and check for feasibility.
    modes = []

    for i in range(0, n_pts + 1):
        for c in lexographic_combinations(n_pts, i):
            if DEBUG:
                print('c', c)
            mask = np.zeros(n_pts, dtype=bool)
            mask[c] = 1
            if DEBUG:
                print('mask', mask)
            C = A[mask, :]
            H = A[~mask, :]

            # Skip case where all contacts active.

            if np.sum(mask) == n_pts:
                check_sliding = True
                null = sp.linalg.null_space(C)
            else:

                # Project into null space.
                if np.sum(mask) > 0:
                    null = sp.linalg.null_space(C)
                    H = np.dot(H, null)
                    if DEBUG:
                        print('projecting H into null space')
                        print(H)


                # Compute interior point.
                b = np.zeros((H.shape[0], 1))
                x = interior_point_halfspace(H, b)
                check_sliding = 1e-5 < np.linalg.norm(np.dot(H, x))
                if check_sliding and len(c) == 0:
                    check_sliding = False
                    m = np.array(['s'] * n_pts)
                    modes.append(m.tolist())

            # If point is strictly interior, then the mode string is valid.
            if check_sliding:
                m = np.array(['s'] * n_pts)
                c = np.array(c)
                m[c] = 'c'
                #modes.append(m.tolist())
                #continue
                # check sliding modes
                # sliding mode enumeration
                sliding_modes = exp_comb(len(c), num_sliding_modes+1) # enumeration sliding sections
                if len(c)==0:
                    modes.append(m.tolist())
                    continue
                for s_mode in sliding_modes:
                    # get sliding matrix T

                    c_sliding = c[s_mode < num_sliding_plane*2]
                    c_line = c[(s_mode >= num_sliding_plane*2) & (s_mode != num_sliding_modes)]
                    mode_sliding = s_mode[s_mode <  num_sliding_plane*2]
                    mode_line = s_mode[(s_mode >= num_sliding_plane*2) & (s_mode != num_sliding_modes)] - 2*num_sliding_plane
                    c_fixed= c[s_mode == num_sliding_modes]
                    Ts = T_sliding[c_sliding,mode_sliding,:,:].reshape(-1,6)
                    T_fixed = np.concatenate(
                        (T_line[c_line, mode_line, :, :].reshape(-1, 6), T[c_fixed, :, :].reshape(-1, 6)))
                    # check interior-
                    if not all(H.shape) :
                        M = np.dot(Ts, null)
                    else:
                        Ts = np.dot(Ts, null)
                        M = np.concatenate((H,Ts))
                    if all(T_fixed.shape) :
                        null_T = sp.linalg.null_space(np.dot(T_fixed,null))
                        if not all(null_T.shape):
                            continue
                        else:
                            M = np.dot(M, null_T)
                    if not all(M.shape):
                        continue


                    b = np.zeros((M.shape[0], 1))
                    x = interior_point_halfspace(-M, b)
                    if 1e-5 < np.linalg.norm(np.dot(M, x)) and all(np.dot(M,x) > 1e-5):
                        s_mode_str = [str(s_mode[i]) for i in range(len(s_mode))]
                        m[c] = s_mode_str
                        modes.append(m.tolist())
                        if DEBUG:
                            print('Appending mode', m.tolist())

    return np.array(sorted(modes))


def enumerate_all_modes_3d(points, normals, tangentials, num_sliding_modes):
    # Check inputs dimensions.
    assert (points.shape[1] == normals.shape[1])
    assert (points.shape[0] == 3)
    assert (normals.shape[0] == 3)
    assert (tangentials.shape[0] == 3)
    assert (tangentials.shape[2] == 2) #  two tangentials vectors: x and y

    # Get dimensions.2
    n_pts = points.shape[1]

    # Create halfspace inequalities, Ax - b ≥ 0.
    n_pts = points.shape[1]
    A = np.zeros((n_pts, 6))
    for i in range(n_pts):
        A[i, 0:3] = normals[:, i].flatten()
        A[i, 3:6] = np.dot(normals[:, i].T, hat(points[:, i])).flatten()

    # Get linearized sliding sections from number of sliding modes
    assert (num_sliding_modes % 2 == 0)
    num_sliding_planes = int(num_sliding_modes/2)
    D = np.array([[np.cos(np.pi*i/num_sliding_planes),np.sin(np.pi*i/num_sliding_planes),0] for i in range(num_sliding_planes)])
    T = np.zeros((n_pts,num_sliding_planes,6)) # sliding plane normals
    for i in range(n_pts):
        R = np.concatenate((tangentials[:, i, :],normals[:, i].reshape(-1,1)), axis=1)
        for j in range(num_sliding_planes):
            T_i = np.dot(R,D[j])
            T[i,j,0:3] = T_i
            T[i,j,3:6] = np.dot(T_i, hat(points[:, i]))

    # keep track for the modes of each face normal
    s_mode_str = [str(i) for i in range(num_sliding_planes)]
    N_modes = np.hstack((np.array(['s']*n_pts),np.matlib.repmat(s_mode_str,1,n_pts).flatten()))
    N = np.vstack((A,T.reshape(-1,T.shape[2])))
    #N = A

    V, Sign = zenotope_vertex(N)# TODO: debug this!!!

    # project V into affine space
    # Project dual points into affine space.
    #null = sp.linalg.null_space((V - V[1,:]))
    orth = sp.linalg.orth((V - V[1,:]).T)
    if orth.shape[1] != V.shape[1]:
        V = np.dot((V-V[1,:]), orth)
    dim_V = V.shape[1]

    vertices = [list(V[i]) for i in range(V.shape[0])]
    ret = pyhull.qconvex('s Fv', vertices)
    print(np.array(ret))

    # get the convex hull of V
    # select faces with desired modes
    # Build facet-vertex incidence matrix.
    n_facets = int(ret[0])
    n_vert = V.shape[0]
    M = np.zeros((n_vert , n_facets), int)
    for i in range(1, len(ret)):
        vert_set = [int(x) for x in ret[i].split(' ')][1:]
        for v in vert_set:
            M[v,i-1] = 1
    if DEBUG:
        print('M')
        print(M)

    # Build face lattice.
    L = FaceLattice(M, dim_V)
    ind_feasible = np.where(np.all(Sign[:,0:A.shape[0]] == 1,axis=1))[0]
    # get feasible mode out from face lattice
    Modes, FeasibleLattice = feasible_faces(L,V,Sign,ind_feasible)
    '''
    Modes_str = []
    for mode in Modes:
        mode_str = ['s'] * n_pts
        for i in range(n_pts):
            m = mode[[i,n_pts+i*2,n_pts+i*2+1]] #TODO: make this generealized
            if m[0] == 1:
                mode_str[i] = 's'
            else:
                mode_str[i] = str(m[1:])
        if not mode_str in Modes_str:
            Modes_str.append(mode_str)
    '''

    return Modes, FeasibleLattice

