# -*- coding: utf-8 -*-

from time import time

import numpy as np
import pyhull
import scipy as sp
from pyhull.halfspace import Halfspace
from scipy.linalg import null_space as null
from scipy.linalg import orth

from contact_modes.geometry import (increment_arrangement, initial_arrangement,
                                    reorder_halfspaces)

from .affine import proj_affine
from .constraints import (build_normal_velocity_constraints,
                          build_tangential_velocity_constraints)
from .helpers import (exp_comb, feasible_faces, get_lattice_mode, hat,
                      lexographic_combinations, signed_covectors,
                      vertex2lattice, zonotope_vertex,unique_row)
from .interior_point import int_pt_cone, interior_point_halfspace
from .lattice import Face, FaceLattice
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

def sample_twist_contact_separating(system, modestr):
    A, b = build_normal_velocity_constraints(system.collider.manifolds)
    c = np.where(modestr == 'c')[0]

    if DEBUG:
        print('cs mode', modestr)
        print('A')
        print(A)

    n_pts = A.shape[0]
    
    mask = np.zeros(n_pts, dtype=bool)
    mask[c] = 1
    n_contacts = np.sum(mask)
    C = A[mask, :]
    H = A[~mask, :]

    n_dim = A.shape[1]
    if n_contacts > n_dim:
        C = orth(C.T @ C).T

    if n_contacts > 0:
        xi = int_pt_cone(H, C, np.zeros((C.shape[0], 1)))
    else:
        xi = int_pt_cone(H)

    if DEBUG:
        print(A @ xi)

    return xi

def sample_twist_sliding_sticking(system, modestr):
    # Create halfspace inequalities, Ax - b ≥ 0.
    print('modestr')
    print(modestr)
    mode = modestr
    if len(mode.shape)==2:
        modestr = mode[0]
        mode = mode[0]
    if DEBUG:
        print(mode)
    
    n_pts = len(system.collider.manifolds)
    num_sliding_planes = int(len(modestr)/n_pts - 1)
    # A = np.zeros((n_pts, 6))
    # for i in range(n_pts):
    #     A[i, 0:3] = normals[:, i].flatten()
    #     A[i, 3:6] = np.dot(normals[:, i].T, hat(points[:, i])).flatten()
    #
    # # Get linearized sliding sections from number of sliding modes
    # D = np.array([[np.cos(np.pi*i/num_sliding_planes),np.sin(np.pi*i/num_sliding_planes),0] for i in range(num_sliding_planes)])
    # T = np.zeros((sum(mode[0:n_pts]==0),num_sliding_planes,6)) # sliding plane normals
    # k=0
    # for i in range(n_pts):
    #     if mode[i] == 1:
    #         continue
    #     R = np.concatenate((tangentials[:, i, :],normals[:, i].reshape(-1,1)), axis=1)
    #     for j in range(num_sliding_planes):
    #         T_i = np.dot(R,D[j])
    #         T[k,j,0:3] = T_i
    #         T[k,j,3:6] = np.dot(T_i, hat(points[:, i]))
    #     k+=1
    A, b = build_normal_velocity_constraints(system.collider.manifolds)
    T, t = build_tangential_velocity_constraints(system.collider.manifolds,num_sliding_planes)
    N = np.vstack((A, T.reshape(-1, T.shape[-1])))

    # identify separation modes
    c_mode = mode[0:n_pts] == 0
    active_mode = np.hstack((np.ones(n_pts,dtype=bool),
                             np.vstack((c_mode, c_mode)).T.flatten()))
    N = N[active_mode]
    # print(active_mode)
    mode = modestr[active_mode]
    # print('mode')
    # print(mode)

    C = N[mode==0]
    H = np.vstack((N[mode==1], -N[mode==-1]))
    # print('H')
    # print(H)

    if all(C.shape):
        # null = sp.linalg.null_space(C)
        # H = np.dot(H, null)
        # x = null @ int_pt_cone(H)
        if C.shape[0] > C.shape[1]:
            C = orth(C.T @ C).T

        x = int_pt_cone(H, C, np.zeros((C.shape[0], 1)))
    else:
        x = int_pt_cone(H)
    
    if DEBUG:
        print('H @ x')
        print(np.dot(H,x))
        print('C @ x')
        print(np.dot(C,x))
        print('x')
        print(x.T)
    return -x

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

def enumerate_contact_separating_3d(A, b):
    # Create solve info.
    info = dict()

    # Create halfspace inequalities, Ax - b <= 0.
    # A, b = build_normal_velocity_constraints(system.collider.manifolds)
    # b = np.zeros(b.shape)
    if DEBUG:
        print('A')
        print(A)
        print('b')
        print(b)

    n_pts = A.shape[0]
    info['n'] = n_pts

    # Get interior point using linear programming.
    t_lp = time()
    int_pt = int_pt_cone(A)
    info['time lp'] = time() - t_lp
    if DEBUG:
        print('int_pt2')
        print(int_pt, '\n', A @ int_pt)

    # Filter contact points which are always in contact.
    mask = np.zeros(n_pts, dtype=bool)
    c = np.where(np.abs(A @ int_pt) < 1e-6)[0] # always contacting.
    mask[c] = 1
    A_c = A[mask,:]
    b_c = b[mask,:]
    A = A[~mask,:]
    b = b[~mask,:]
    if DEBUG:
        print(c)
        print(mask)

    # Project into null space of contacting points.
    # [A'; A_c]x ≤ 0, A_c⋅x = 0 ⇒ x ∈ NULL(A_c)
    # let NULL(A_c) = [x0, ... , xc]
    # A'⋅NULL(A_c)x' ≤ 0
    if np.sum(mask) > 0:
        N = null(A_c, np.finfo(np.float32).eps)
        A = A @ N
        int_pt = np.linalg.lstsq(N, int_pt, None)[0]
        if DEBUG:
            print('Null A_c')
            print(N)
            print('new int pt')
            print(int_pt)
            print(A @ int_pt)

    # Compute dual points.
    b_off = b - np.dot(A, int_pt)
    dual = A / b_off
    if DEBUG:
        print('b off')
        print(b_off)
        print('dual')
        print(dual)

    # Handle degenerate cases when d = 0 or 1.
    if np.sum(mask) == n_pts:
        cs_modes = [np.array(['c']*n_pts)]
        lattice = FaceLattice()
        lattice.L = [[Face(range(n_pts), 0)]]
        lattice.L[0][0].m = cs_modes[0]
        return cs_modes, lattice, info
    if dual.shape[1] == 1:
        lattice = FaceLattice(M=np.ones((1,1), int), d=1)
        dual_map = [list(np.where(~mask)[0])]
        cs_modes = lattice.csmodes(mask, dual_map)
        lattice.L = lattice.L[1:3]
        return cs_modes[1:3], lattice, info

    # Project dual points into affine space.
    dual = proj_affine(dual.T).T
    if DEBUG:
        print('proj dual')
        print(dual)
    info['d'] = dual.shape[1]

    # Filter duplicate points.
    idx = np.lexsort(np.rot90(dual))
    dual_map = []
    dual_unique = []
    i = 0
    while i < len(idx):
        if i == 0:
            dual_unique.append(dual[idx[i],:])
            dual_map.append([idx[i]])
        else:
            curr = dual[idx[i], :]
            last = dual_unique[-1]
            if np.linalg.norm(last - curr) < 1e-6:
                dual_map[-1].append(idx[i])
            else:
                dual_unique.append(dual[idx[i],:])
                dual_map.append([idx[i]])
        i += 1
    if DEBUG:
        print('dual map')
        print(dual_map)
        print('dual unique')
        print(dual_unique)

    # Handle various cases.
    dual = [list(dual_unique[i]) for i in range(len(dual_unique))]
    if len(dual_unique) == 1:
        d = 0
        M = np.array([[]])
    elif len(dual_unique[0]) == 1:
        d = 1
        M = np.zeros((len(dual), 2), int)
        i_min = np.argmin(np.array(dual).flatten())
        i_max = np.argmax(np.array(dual).flatten())
        M[i_min,0] = 1
        M[i_max,1] = 1
    else:
        d = len(dual[0])

        # Compute dual convex hull.
        t_start = time()
        ret = pyhull.qconvex('Fv', dual)
        info['time conv'] = time() - t_start

        if DEBUG:
            print(np.array(ret))

        # Build facet-vertex incidence matrix.
        n_facets = int(ret[0])
        M = np.zeros((len(dual), n_facets), int)
        for i in range(1, len(ret)):
            vert_set = [int(x) for x in ret[i].split(' ')][1:]
            for v in vert_set:
                M[v,i-1] = 1

    if DEBUG:
        print('dual')
        print(np.array(dual))

    if DEBUG:
        print('M')
        print(M)

    # Build face lattice.
    t_start = time()
    lattice = FaceLattice(M, d)
    info['time lattice'] = time() - t_start

    # Build mode strings.
    cs_modes = lattice.csmodes(mask, dual_map)

    if DEBUG:
        print(cs_modes)

    info['# 0 faces'] = lattice.num_k_faces(0)
    info['# d-1 faces'] = lattice.num_k_faces(info['d']-1)
    info['# faces'] = lattice.num_faces()

    # Return mode strings.
    return cs_modes, lattice, info

def enum_sliding_sticking_3d_proj(A, b, T, bt):
    info = dict()
    info['time zono'] = 0
    info['time lattice'] = 0

    cs_modes, cs_lattice, cs_info = enumerate_contact_separating_3d(A, b)
    all_modes = []
    #
    # manifolds = system.collider.manifolds
    # n_contacts = len(manifolds)
    # points = np.zeros((3,n_contacts))
    # tangents = np.zeros((3,n_contacts,2))
    # normals = np.zeros((3,n_contacts))
    # for i in range(n_contacts):
    #     m = manifolds[i]
    #     body_A = m.shape_A
    #
    #     if body_A.num_dofs() > 0:
    #
    #         g_wo = body_A.get_transform_world()
    #         g_wc = m.frame_A()
    #         g_oc = SE3.inverse(g_wo) * g_wc
    #         g_oc_m = g_oc.matrix()
    #         print(g_oc_m)
    #         points[:,i] = g_oc_m[1:3,-1]
    #         normals[:, i] = g_oc_m[1:3, 2]
    #         tangents[:, i] = g_oc_m[1:3, 0:1]
    # print(points)
    # print(normals)
    # A, b = build_normal_velocity_constraints(system.collider.manifolds)
    # T, bt = build_tangential_velocity_constraints(system.collider.manifolds, num_sliding_planes)
    n_pts = A.shape[0]
    num_sliding_planes = int(T.shape[0] / n_pts)
    # A_, b_ = contacts_to_half(points, normals)
    # print('A')
    # print(A)
    # print('A_')
    # print(A_)

    #Get linearized sliding sections from number of sliding modes
    # D = np.array([[np.cos(np.pi*i/num_sliding_planes),np.sin(np.pi*i/num_sliding_planes),0] for i in range(num_sliding_planes)])
    # T = np.zeros((n_pts,num_sliding_planes,6)) # sliding plane normals
    # for i in range(n_pts):
    #     R = np.concatenate((tangents[:, i, :],normals[:, i].reshape(-1,1)), axis=1)
    #     for j in range(num_sliding_planes):
    #         T_i = np.dot(R,D[j])
    #         T[i,j,0:3] = T_i
    #         T[i,j,3:6] = np.dot(T_i, hat(points[:, i]))
    # T *= -1

    H = np.vstack((A, T.reshape(-1, T.shape[-1])))

    num_modes = 0
    for layer in cs_lattice.L:
        for face in layer:
            cs_mode = face.m
            print(cs_mode)
            mask_c = cs_mode == 'c'
            mask_s = ~mask_c
            # mask = np.hstack((mask_s, np.array([mask_c] * num_sliding_planes).T.flatten()))
            mask = np.hstack((cs_mode=='0', np.array([mask_c] * num_sliding_planes).T.flatten()))
            if all(mask_s):

                L = FaceLattice()
                L.L=[]
                L.append_empty()
                mode_sign = np.hstack((np.ones(n_pts,dtype=int),np.zeros(n_pts*num_sliding_planes,dtype=int)))
                modes = [mode_sign]
                L.L[0][0].m = mode_sign

            else:
                nc = null(A[mask_c], np.finfo(np.float32).eps)

                if not np.all(nc.shape):
                    mode_sign = np.zeros(H.shape[0])
                    L = FaceLattice()
                    L.L = []
                    L.append_empty()
                    L.L[0][0].m = [mode_sign]
                    modes = [mode_sign]
                elif nc.shape[1] == 1: # only able to move in 1 dim
                    move_direc = np.array([nc,-nc,np.zeros(nc.shape)])
                    vd = np.dot(H, move_direc)
                    vd[abs(vd)<1e-6] = 0
                    mode_sign = np.sign(vd).T.squeeze()
                    mode_cs_sign = mode_sign[:, 0:n_pts]
                    feasible_ind = np.all(mode_cs_sign[:, mask_s] == 1, axis=1)
                    mode_sign = mode_sign[feasible_ind]
                    L = FaceLattice()
                    L.L = []
                    L.append_empty()
                    L.L[0][0].m = mode_sign
                    modes = mode_sign
                else:

                    H_proj = np.dot(H[mask], nc)
                    H_proj_u,idu = unique_row(H_proj)
                    # if H_proj_u.shape[0] != H_proj.shape[0]:
                    #     print('AHH')
                    # print(H_proj_u)
                    t_start = time()
                    V_all, Sign_all = zonotope_vertex(H_proj_u)
                    if len(V_all) == 0:
                        continue

                    #print(Sign_all)
                    info['time zono'] += time() - t_start
                    Sign_all = Sign_all[:,idu]
                    feasible_ind = np.where(np.all(Sign_all[:, 0:sum(mask_s)] == 1, axis=1))[0]
                    V = V_all[feasible_ind]
                    Sign = Sign_all[feasible_ind]
                    #print(feasible_ind)
                    if not np.all(feasible_ind.shape):
                        continue
                    V_uq,ind_uq = unique_row(V)
                    if not V_uq.shape[0] == V.shape[0]:
                        Sign_uq = np.zeros((V_uq.shape[0], Sign.shape[1]))
                        for i in range(V_uq.shape[0]):
                            s_all = Sign[ind_uq == i]
                            s = np.zeros((1,Sign.shape[1]))
                            if s_all.shape[0]>1:
                                s[:,np.all(s_all == 1, axis = 0)] = 1
                                s[:,np.all(s_all == -1, axis = 0)] = -1
                            Sign_uq[i] = s
                        V = V_uq
                        Sign = Sign_uq

                    # Hack.
                    V = V_all
                    Sign = Sign_all

                    # sign_cells = I.sign_vectors(I.dim(), I0)
                    # idx_sc = np.where(np.all(sign_cells[:, 0:sum(mask_s)] == 1, axis=1))[0]
                    # sign_cells = sign_cells[idx_sc]
                    # idx_sc = np.lexsort(np.rot90(sign_cells))
                    # sign_cells = sign_cells[idx_sc]
                    # print(sign_cells)
                    # print('# v', len(V_all))
                    # idx_s = np.lexsort(np.rot90(Sign))
                    # print(Sign[idx_s])

                    t_start = time()
                    L = vertex2lattice(V)
                    info['time lattice'] += time() - t_start

                    mode_sign = np.zeros((Sign.shape[0],n_pts*(1+num_sliding_planes)))
                    mode_sign[:,mask] = Sign
                    modes = get_lattice_mode(L,mode_sign)

                    print('# modes', len(modes))
            print(np.array(modes))

            num_modes+=len(modes)
            all_modes.append(modes)
            face.ss_lattice = L

    info['# faces'] = num_modes
    
    print(num_modes)

    # np.set_printoptions(suppress=True)
    # for cs_layer in cs_lattice.L:
    #     for cs_face in cs_layer:
    #         for ss_layer in cs_face.ss_lattice.L:
    #             for ss_face in ss_layer:
    #                 if hasattr(ss_face, 'm'):
    #                     print(sample_twist_sliding_sticking(points, normals, tangentials, ss_face.m).T)
    return all_modes, cs_lattice, info
