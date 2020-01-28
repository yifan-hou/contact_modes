# -*- coding: utf-8 -*-

from time import time

import numpy as np
import pyhull
import scipy as sp
from pyhull.halfspace import Halfspace

from .helpers import hat, lexographic_combinations, exp_comb, zenotope_vertex,\
    feasible_faces, vertex2lattice, get_lattice_mode, zenotope_add, to_lattice


from scipy.linalg import null_space as null

from .interior_point import int_pt_cone, interior_point_halfspace
from .lattice import FaceLattice
from .se3 import *

DEBUG = False

def make_frame(z):
    z = z.reshape((3,1))
    z = z / np.linalg.norm(z)
    n = null(z.T)
    x = n[:,0,None]
    x = x / np.linalg.norm(x)
    y = np.cross(z, x, axis=0)
    y = y / np.linalg.norm(y)
    R = np.zeros((3,3), dtype='float32')
    R[:,0,None] = x
    R[:,1,None] = y
    R[:,2,None] = z
    return R

def contacts_to_half(points, normals):
    # n_pts = points.shape[1]
    # A = np.zeros((n_pts, 6))
    # for i in range(n_pts):
    #     g_oc = SE3()
    #     g_oc.set_rotation(make_frame(normals[:,i,None]))
    #     g_oc.set_translation(points[:,i,None])
    #     Ad = SE3.Ad(SE3.inverse(g_oc))
    #     B = np.array([0, 0, 1., 0, 0, 0]).reshape((6,1))
    #     A[i,:] = B.T @ Ad
    # # print(A)

    # Create halfspace inequalities, Ax - b ≥ 0.
    n_pts = points.shape[1]
    A = np.zeros((n_pts, 6))
    for i in range(n_pts):
        A[i,0:3] = normals[:,i].flatten()
        A[i,3:6] = np.dot(normals[:,i].T, -hat(points[:,i])).flatten()
    A *= -1
    b = np.zeros((n_pts, 1))
    # print(A)

    return A, b

def sample_twist_contact_separating(points, normals, modestr):
    A, b = contacts_to_half(points, normals)
    c = np.where(modestr == 'c')[0]

    n_pts = points.shape[1]
    mask = np.zeros(n_pts, dtype=bool)
    mask[c] = 1
    n_contacts = np.sum(mask)
    C = A[mask, :]
    H = A[~mask, :]

    if n_contacts > 0:
        xi = int_pt_cone(H, C, np.zeros((n_contacts, 1)))
    else:
        xi = int_pt_cone(H)

    if DEBUG:
        print(A @ xi)

    return xi

def sample_twist_sliding_sticking(points, normals, tangentials, modestr):
    # Create halfspace inequalities, Ax - b ≥ 0.
    mode = modestr
    n_pts = points.shape[1]
    num_sliding_planes = int(len(modestr)/n_pts - 1)
    A = np.zeros((n_pts, 6))
    for i in range(n_pts):
        A[i, 0:3] = normals[:, i].flatten()
        A[i, 3:6] = np.dot(normals[:, i].T, hat(points[:, i])).flatten()

    # Get linearized sliding sections from number of sliding modes
    D = np.array([[np.cos(np.pi*i/num_sliding_planes),np.sin(np.pi*i/num_sliding_planes),0] for i in range(num_sliding_planes)])
    T = np.zeros((sum(mode[0:n_pts]==0),num_sliding_planes,6)) # sliding plane normals
    k=0
    for i in range(n_pts):
        if mode[i] == 1:
            continue
        R = np.concatenate((tangentials[:, i, :],normals[:, i].reshape(-1,1)), axis=1)
        for j in range(num_sliding_planes):
            T_i = np.dot(R,D[j])
            T[k,j,0:3] = T_i
            T[k,j,3:6] = np.dot(T_i, hat(points[:, i]))
        k+=1

    # identify separation modes
    c_mode = mode[0:n_pts] == 0
    active_mode = np.hstack((np.ones(n_pts,dtype=bool),np.vstack((c_mode, c_mode)).T.flatten()))
    mode = modestr[active_mode]

    N = -np.vstack((A, T.reshape(-1, T.shape[2])))
    C = N[mode==0]
    H = np.vstack((N[mode==1], -N[mode==-1]))

    if all(C.shape):
        # null = sp.linalg.null_space(C)
        # H = np.dot(H, null)
        # x = null @ int_pt_cone(H)
        x = int_pt_cone(H, C, np.zeros((C.shape[0], 1)))
    else:
        x = int_pt_cone(H)
    #print(np.dot(N,x))
    return x

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

def enum_sliding_sticking_3d(points, normals, tangentials, num_sliding_planes):

    cs_modes, cs_lattice = enumerate_contact_separating_3d(points, normals)
    all_modes = []
    n_pts = points.shape[1]
    A, b = contacts_to_half(points, normals)
    # Get linearized sliding sections from number of sliding modes
    D = np.array([[np.cos(np.pi*i/num_sliding_planes),np.sin(np.pi*i/num_sliding_planes),0] for i in range(num_sliding_planes)])
    T = np.zeros((n_pts,num_sliding_planes,6)) # sliding plane normals
    for i in range(n_pts):
        R = np.concatenate((tangentials[:, i, :],normals[:, i].reshape(-1,1)), axis=1)
        for j in range(num_sliding_planes):
            T_i = np.dot(R,D[j])
            T[i,j,0:3] = T_i
            T[i,j,3:6] = np.dot(T_i, hat(points[:, i]))
    T *= -1
    H = np.vstack((A, T.reshape(-1, T.shape[2])))

    num_modes = 0
    for layer in cs_lattice.L:
        for face in layer:
            cs_mode = face.m
            mask_c = cs_mode == 'c'
            mask_s = ~mask_c
            mask = np.hstack((mask_s, np.array([mask_c] * num_sliding_planes).T.flatten()))
            if all(mask_s):

                L = FaceLattice()
                L.L=[]
                L.append_empty()
                mode_sign = np.hstack((np.ones(n_pts,dtype=int),np.zeros(n_pts*num_sliding_planes,dtype=int)))
                modes = [mode_sign]
                L.L[0][0].m = mode_sign

            else:

                V_all, Sign_all = zenotope_vertex(H[mask])
                feasible_ind = np.where(np.all(Sign_all[:, 0:sum(mask_s)] == 1, axis=1))[0]
                V = V_all[feasible_ind]
                Sign = Sign_all[feasible_ind]
                L = vertex2lattice(V)

                mode_sign = np.zeros((Sign.shape[0],n_pts*(1+num_sliding_planes)))
                mode_sign[:,mask] = Sign
                modes = get_lattice_mode(L,mode_sign)

            num_modes+=len(modes)
            all_modes.append(modes)
            face.ss_lattice = L
    print(num_modes)

    return all_modes, cs_lattice

def enum_sliding_sticking_3d_incremental(points, normals, tangentials, num_sliding_planes):

    cs_modes, cs_lattice = enumerate_contact_separating_3d(points, normals)
    all_modes = []
    n_pts = points.shape[1]
    A, b = contacts_to_half(points, normals)
    # Get linearized sliding sections from number of sliding modes
    D = np.array([[np.cos(np.pi*i/num_sliding_planes),np.sin(np.pi*i/num_sliding_planes),0] for i in range(num_sliding_planes)])
    T = np.zeros((n_pts,num_sliding_planes,6)) # sliding plane normals
    for i in range(n_pts):
        R = np.concatenate((tangentials[:, i, :],normals[:, i].reshape(-1,1)), axis=1)
        for j in range(num_sliding_planes):
            T_i = np.dot(R,D[j])
            T[i,j,0:3] = T_i
            T[i,j,3:6] = np.dot(T_i, hat(points[:, i]))
    T *= -1
    H = np.vstack((A,T.reshape(-1,T.shape[2])))
    Vertices = dict()
    Signs = dict()
    masks = []
    num_modes = 0
    for layer in cs_lattice.L:
        for face in layer:
            cs_mode = face.m
            mask_c = cs_mode == 'c'
            mask_s = ~mask_c
            mask = np.hstack((mask_s,np.array([mask_c]*num_sliding_planes).T.flatten()))
            if all(mask_s):

                L = FaceLattice()
                L.L=[]
                L.append_empty()
                mode_sign = np.hstack((np.ones(n_pts,dtype=int),np.zeros(n_pts*num_sliding_planes,dtype=int)))
                modes = [mode_sign]
                L.L[0][0].m = mode_sign

            else:
                # As = A[mask_s]
                # Tc = T[mask_c].reshape(-1,T.shape[2])
                # N = np.vstack((As,Tc))
                if len(masks) == 0:
                    V_all, Sign_all = zenotope_vertex(H[mask])
                else:
                    mask_ = []
                    for m in masks:
                        if np.all(np.isin(np.where(m)[0],np.where(mask)[0])) and sum(m) > sum(mask_):
                            mask_ = m
                    if len(mask_) == 0:
                        V_all, Sign_all = zenotope_vertex(H[mask])
                    else:
                        add_normals = H[mask_!=mask]
                        V_all, Sign_all = zenotope_add(Vertices[str(mask_)],Signs[str(mask_)], add_normals)
                masks.append(mask)
                Vertices[str(mask)] = V_all
                Signs[str(mask)] = Sign_all
                # V_all, Sign_all = zenotope_vertex(N)
                V = V_all[np.all(Sign_all[:, 0:sum(mask_s)] == 1, axis=1)]
                Sign = Sign_all[np.all(Sign_all[:, 0:sum(mask_s)] == 1, axis=1)]

                L = vertex2lattice(V)

                mode_sign = np.zeros((Sign.shape[0],n_pts*(1+num_sliding_planes)))
                mode_sign[:,mask] = Sign
                modes = get_lattice_mode(L,mode_sign)

            num_modes+=len(modes)
            all_modes.append(modes)
            face.ss_lattice = L
    print(num_modes)

    return all_modes, cs_lattice


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

    N = -np.vstack((A,T.reshape(-1,T.shape[2])))
    #N = A

    V, Sign = zenotope_vertex(N)

    L = vertex2lattice(V)
    ind_feasible = np.where(np.all(Sign[:,0:A.shape[0]] == 1,axis=1))[0]
    # get feasible mode out from face lattice
    Modes, FeasibleLattice = feasible_faces(L,V,Sign,ind_feasible)
    # filter lattice
    Modes_str = []
    Modes_dict = dict()
    FilteredLattice = FaceLattice()
    FilteredLattice.L = []
    for layer in FeasibleLattice.L:
        FilteredLattice.L.append([])
        for face in layer:
            mode_str = ['s'] * n_pts
            mode = face.m
            for i in range(n_pts):
                m = mode[[i,n_pts+i*num_sliding_planes,n_pts+i*num_sliding_planes+1]]
                if m[0] == 1:
                    mode_str[i] = 's'
                else:
                    mode_str[i] = str(m[1:])
            if not mode_str in Modes_str:
                Modes_str.append(mode_str)
                Modes_dict[str(mode_str)] = face
                face.m_str = mode_str
                face.modes = [face.m]
                FilteredLattice.L[-1].append(face)
            else:
                Modes_dict[str(mode_str)].verts += face.verts
                Modes_dict[str(mode_str)].parents += face.parents
                Modes_dict[str(mode_str)].modes += [face.m]
                #layer.remove(face)

    return Modes_str, FilteredLattice
