
# -*- coding: utf-8 -*-

from time import time

import numpy as np
import pyhull
import scipy as sp
from pyhull.halfspace import Halfspace
from .helpers import hat, lexographic_combinations, exp_comb, zenotope_vertex
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


def enumerate_all_modes_3d_exponential(points, normals, tangentials, num_sliding_modes):
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

    # Get linearized sliding sections from number of sliding modes
    assert (num_sliding_modes % 2 == 0)
    D = np.array([[np.sin(2*np.pi*i/num_sliding_modes),np.cos(2*np.pi*i/num_sliding_modes),0] for i in range(num_sliding_modes)])
    # Get sliding mode section enumeration
    sliding_sections = np.identity(num_sliding_modes) - np.roll(np.identity(num_sliding_modes),1,axis=1)
    T = np.zeros((n_pts,num_sliding_modes,2,6))
    T_fixed = np.zeros((n_pts,4,6))
    for i in range(n_pts):
        D_i = np.concatenate((tangentials[:,i,:].T,-tangentials[:,i,:].T))
        T_i = [np.dot(normals[:, i].T, hat(t)) for t in D_i]
        T_fixed[i, :, 0:3] = T_i
        T_fixed[i, :, 3:6] = np.dot(T_i, hat(points[:, i]))
        for j in range(num_sliding_modes):
            section = sliding_sections[j]
            D_i = (section*D.T).T
            D_i = D_i[section!=0]
            T_i = [np.dot(normals[:,i].T,hat(t)) for t in D_i]
            T[i,j,:,0:3] = T_i
            T[i, j, :, 3:6] = np.dot(T_i, hat(points[:, i]))

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

            # If point is strictly interior, then the mode string is valid.
            if check_sliding:
                m = np.array(['s'] * n_pts)
                c_normals = normals[:,c]
                c_points = points[:,c]
                c = np.array(c)
                # check sliding modes
                # sliding mode enumeration
                sliding_modes = exp_comb(len(c), num_sliding_modes+1) # enumeration sliding sections
                if len(c)==0:
                    modes.append(m.tolist())
                    continue
                for s_mode in sliding_modes:
                    # get sliding matrix T

                    c_sliding = c[s_mode < num_sliding_modes]
                    mode_sliding = s_mode[s_mode < num_sliding_modes]
                    c_fixed= c[s_mode == num_sliding_modes]
                    Ts = np.concatenate((T[c_sliding,mode_sliding,:,:].reshape(-1,6),T_fixed[c_fixed,:,:].reshape(-1,6)))
                    # check interior-
                    if len(H) == 0:
                        M = Ts
                    else:
                        Ts = np.dot(Ts, null)
                        M = np.concatenate((H,Ts))
                    b = np.zeros((M.shape[0], 1))
                    x = interior_point_halfspace(M, b)
                    if 1e-5 < np.linalg.norm(np.dot(M, x)):
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
    D = np.array([[np.sin(2*np.pi*i/num_sliding_modes),np.cos(2*np.pi*i/num_sliding_modes),0] for i in range(num_sliding_modes)])
    T = np.zeros((n_pts,num_sliding_modes,6)) # sliding plane normals
    for i in range(n_pts):
        R = np.concatenate((normals[:, i].reshape(-1,1), tangentials[:, i, :]), axis=1)
        for j in range(num_sliding_modes):
            T_i = np.dot(np.linalg.inv(R),D[j])
            T[i,j,0:3] = T_i
            T[i,j,3:6] = np.dot(T_i, hat(points[:, i]))

    # keep track for the modes of each face normal
    s_mode_str = [str(i) for i in range(num_sliding_modes)]
    N_modes = np.hstack((np.array(['s']*n_pts),np.matlib.repmat(s_mode_str,1,n_pts).flatten()))
    N = np.vstack((A,T.reshape(-1,T.shape[2])))

    V, Sign = zenotope_vertex(N)# TODO: debug this!!!

    vertices = [list(V[i]) for i in range(V.shape[0])]
    ret = pyhull.qconvex('s Fv', vertices)
    print(np.array(ret))

    # get the convex hull of V
    # select faces with desired modes
    return ret
