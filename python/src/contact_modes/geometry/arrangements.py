from time import time
import operator as op
from functools import reduce

import numpy as np
from scipy.linalg import null_space

from contact_modes import lexographic_combinations

from .incidence_graph import *


DEBUG=False
PROFILE=False

def num_incidences_simple(d):
    i_sum = 0
    for i in range(0, d+1):
        f_i = 0
        for j in range(0, i+1):
            f_i += ncr(d-j, i-j) * ncr(d, d-j)
        i_sum += i * f_i + 2*(d-i) * f_i
    return i_sum

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def get_sign(v, eps=np.finfo(np.float32).eps):
    v = v.flatten()
    return np.asarray(np.sign(np.where(np.abs(v) > eps, v, np.zeros((v.shape[0],)))), int)

def vector(e):
    if len(e.subfaces) == 2:
        v0 = e.subfaces[0]
        v1 = e.subfaces[1]
        return v1.int_pt - v0.int_pt
    elif len(e.subfaces) == 1:
        v0 = e.subfaces[0]
        return e.int_pt - v0.int_pt
    else:
        assert(False)

def sign_vertex(v, a, b, eps=np.finfo(np.float32).eps):
    return get_sign(a@v.int_pt - b, eps).item()

def color_vertex(v, a, b, eps=np.finfo(np.float32).eps):
    s = sign_vertex(v, a, b, eps)
    if s == 0:
        return COLOR_AH_CRIMSON
    else:
        return COLOR_AH_WHITE

def color_edge(e, a, b, eps=np.finfo(np.float32).eps):
    # Assume |a| = 1.
    if len(e.subfaces) == 2:
        v0 = e.subfaces[0]
        v1 = e.subfaces[1]
        s0 = sign_vertex(v0, a, b, eps)
        s1 = sign_vertex(v1, a, b, eps)
        if s0 * s1 == 1:
            return COLOR_AH_WHITE
        elif s0 == 0 and s1 == 0:
            return COLOR_AH_CRIMSON
        elif s0 == 0 or s1 == 0:
            return COLOR_AH_PINK
        elif s0 + s1 == 0:
            return COLOR_AH_RED
        else:
            assert(False)
    elif len(e.subfaces) == 1:
        v0 = e.subfaces[0]
        s0 = sign_vertex(v0, a, b, eps)
        v_e = vector(e)
        s_e = get_sign(a@v_e, eps).item()
        if s0 == 0 and s_e == 0:
            return COLOR_AH_CRIMSON
        elif s0 == 0 and s_e != 0:
            return COLOR_AH_PINK
        elif s0 * s_e == 1:
            return COLOR_AH_WHITE
        elif s0 * s_e == -1:
            return COLOR_AH_RED
        elif s0 != 0 and s_e == 0:
            return COLOR_AH_WHITE
        # TODO Check above conditions.
        else:
            assert(False)
    else:
        assert(False)

def reorder_halfspaces(A, b, eps=np.finfo(np.float32).eps):
    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    d = A.shape[1]
    I = list(range(n))
    i = 1
    j = 1
    while i < n and j < d:
        A0 = np.concatenate((A[0:j], A[i,None]), axis=0)
        if np.linalg.matrix_rank(A0, eps) > j:
            A[j], A[i] = A[i].copy(), A[j].copy()
            b[j], b[i] = b[i], b[j]
            I[j], I[i] = I[i], I[j]
            j += 1
        i += 1
    return A, b, np.array(I, int)

def initial_arrangement(A, b, eps=np.finfo(np.float32).eps):
    # Assert we are given d linearly independent hyperplanes.
    n = A.shape[0]
    d = A.shape[1]
    assert(n == d)
    if DEBUG:
        print(A)
        print('rank(A)', np.linalg.matrix_rank(A, eps))
        print('d', d)
    assert(np.linalg.matrix_rank(A, eps) == d)

    # Build faces from top to bottom.
    I = IncidenceGraph(d)
    I.add_halfspace(A, b)
    # d+1 face
    one = Node(d+1)
    I.add_node(one)
    # d faces
    for i in range(d+1):
        for c in lexographic_combinations(d, i):
            f = Node(d)
            f._sv_key = np.ones((d,), int)
            f._sv_key[c] = -1
            f.superfaces.append(one)
            one.subfaces.append(f)
            I.add_node(f)
    # TODO check # incidences for rank d nodes
    # TODO check # incidences for rank d+1 node
    # k faces, 0 ≤ k ≤ d-1
    for k in range(d-1, -1, -1):
        R_k = I.rank(k + 1)
        for g in R_k.values():
            for i in range(d):
                sign_vector = g._sv_key.copy()
                if sign_vector[i] == 0:
                    continue
                sign_vector[i] = 0
                f = I.find(sv=sign_vector, r=k)
                if f is None:
                    f = Node(k)
                    f._sv_key = sign_vector
                    I.add_node(f)
                f.superfaces.append(g)
                g.subfaces.append(f)
            if DEBUG:
                assert(len(g.subfaces) == k+1)
                if k+1 != d:
                    assert(len(g.superfaces) == 2*(d-k-1))
    # TODO check # incidences for rank 0 node
    # -1 face
    zero = Node(-1)
    vertex = I.find(sv=np.zeros((d,),int), r=0)
    zero.superfaces.append(vertex)
    vertex.subfaces.append(zero)
    I.add_node(zero)

    # Computer interior point for 0 face.
    vertex.int_pt = np.linalg.solve(A, b)

    # Compute interior point for 1 faces.
    for f in I.rank(1).values():
        null = null_space(A[f._sv_key == 0])
        dot = A[f._sv_key != 0] @ null
        i = np.where(f._sv_key != 0)[0][0]
        if dot * f._sv_key[i] > 0:
            f.int_pt = vertex.int_pt + null
        else:
            f.int_pt = vertex.int_pt - null
        if DEBUG:
            # print(' f.sv', f._sv_key)
            pt = (A @ f.int_pt - b).flatten().T
            sv = get_sign(pt)
            assert(np.array_equal(f._sv_key, sv))
            # print(get_sign(pt))
            # print('A@f-b', (A @ f.int_pt - b).flatten().T)

    # Computer interior point for k faces, 2 ≤ k ≤ d
    for k in range(2, d+1):
        for f in I.rank(k).values():
            pts = np.array([h.int_pt.flatten() for h in f.subfaces]).T
            f.int_pt = np.mean(pts, axis=1, keepdims=True)
            if DEBUG:
                pt = (A @ f.int_pt - b).flatten().T
                sv = get_sign(pt)
                assert(np.array_equal(f._sv_key, sv))
                # print(' f.sv', f._sv_key)
                # print('A@f-b', (A @ f.int_pt - b).flatten().T)

    return I

def increment_arrangement(a, b, I, eps=np.finfo(np.float32).eps):
    # Normalize halfspace, |a| = 1.
    a = np.reshape(a, (1,-1))
    b = np.reshape(b, (1, 1))
    norm_a = np.linalg.norm(a)
    a = a / norm_a
    b = b / norm_a
    I.add_halfspace(a, b)
    # ==========================================================================
    # Phase 1: Find an edge e₀ in 𝓐(H) such that cl(e₀) ∩ h ≠ ∅
    # ==========================================================================
    if DEBUG:
        print('PHASE 1')
    n = I.num_halfspaces()
    u = I.get(0, 0)
    # Find an incident edge e on v such that aff(e) is not parallel to h.
    for e in u.superfaces:
        v_e = vector(e)
        dist = np.linalg.norm(a @ v_e) # TODO Check this
        if dist > eps:
            break
    if DEBUG:
        assert(dist > eps)
    # Find edge e₀ such that cl(e₀) ∩ h ≠ ∅.
    e0 = e
    v_e0 = vector(e0) / np.linalg.norm(vector(e0))
    while True:
        if color_edge(e0, a, b, eps) > COLOR_AH_WHITE:
            # if DEBUG:
            #     print(e0._sv_key)
            #     print('color', color_edge(e0, a, b, eps))
            break
        # Find v(e0) closer to h.
        if len(e0.subfaces) == 2:
            v0 = e0.subfaces[0]
            v1 = e0.subfaces[1]
            d0 = np.abs(a @ v0.int_pt - b)
            d1 = np.abs(a @ v1.int_pt - b)
            if d0 < d1:
                v = v0
            else:
                v = v1
        if len(e0.subfaces) == 1:
            v = e0.subfaces[0]
        # Find e in v such that aff(e0) == aff(e).
        e_min = None
        v_min = None
        min_dist = np.inf
        for e in v.superfaces:
            if e is e0:
                continue
            v_e = vector(e) / np.linalg.norm(vector(e))
            dist = np.linalg.norm(v_e - (v_e0.T @ v_e) * v_e0)
            if dist < min_dist:
                e_min = e
                v_min = v_e
                min_dist = dist
        e0 = e_min
        # if DEBUG:
        #     print('e0', color_edge(e0, a, b, eps))
        #     print('e0', e0._sv_key.astype(float))
    
    # ==========================================================================
    # Phase 2: Mark all faces f with cl(f) ∩ h ≠ ∅ pink, red, or crimson.
    # ==========================================================================
    if DEBUG:
        print('PHASE 2')
    # Add some 2 face incident upon e₀ to Q and mark it green.
    f = e0.superfaces[0]
    f.color = COLOR_AH_GREEN
    Q = [f]
    # Color vertices, edges, and 2 faces of 𝓐(H).
    d = a.shape[1]
    L = [[] for i in range(d+1)]
    while Q:
        f = Q.pop()
        for e in f.subfaces:
            if e.color != COLOR_AH_WHITE:
                continue
            color_e = color_edge(e, a, b, eps)
            if color_e > COLOR_AH_WHITE:
                # Mark each white vertex v ∈ h crimson and insert v into L₀.
                for v in e.subfaces:
                    if v.color == COLOR_AH_WHITE:
                        color_v = color_vertex(v, a, b, eps)
                        if color_v == COLOR_AH_CRIMSON:
                            v.color = color_v
                            L[0].append(v)
                # Color e and insert e into L₁.
                e.color = color_e
                L[1].append(e)
                # Mark all white 2 faces green and put them into Q.
                for g in e.superfaces:
                    if g.color == COLOR_AH_WHITE:
                        g.color = COLOR_AH_GREEN
                        Q.append(g)
    # Color k faces, 2 ≤ k ≤ d.
    for k in range(2, d+1):
        for f in L[k-1]:
            for g in f.superfaces:
                if g.color != COLOR_AH_WHITE and g.color != COLOR_AH_GREEN:
                    continue

                if f.color == COLOR_AH_PINK:
                    if DEBUG and k == d:
                        # print('f  ', f._sv_key.astype(float))
                        # print('g  ', g._sv_key.astype(float))
                        # print('g h', get_sign(I.A @ g.int_pt - I.b, eps).astype(float))
                        pass
                    above = 0
                    below = 0
                    for f_g in g.subfaces:
                        if f_g.color == COLOR_AH_RED:
                            above = 1
                            below = 1
                            break
                        if f_g._sign_bit_n != n:
                            s = get_sign(a @ f_g.int_pt - b, eps)
                            f_g._sign_bit_n = n
                            f_g._sign_bit = s
                        else:
                            s = f_g._sign_bit
                        # if DEBUG:
                        #     print('f_g', s.astype(float))
                        if s > 0:
                            above = 1
                        elif s < 0:
                            below = 1
                    if above * below == 1:
                        g.color = COLOR_AH_RED
                    else:
                        g.color = COLOR_AH_PINK
                
                elif f.color == COLOR_AH_RED:
                    g.color = COLOR_AH_RED

                elif f.color == COLOR_AH_CRIMSON:
                    crimson = True
                    for f_g in g.subfaces:
                        if f_g.color != COLOR_AH_CRIMSON:
                            crimson = False
                            break
                    if crimson:
                        g.color = COLOR_AH_CRIMSON
                    else:
                        g.color = COLOR_AH_PINK
                
                else:
                    # print('f sv', f._sv_key.astype(float))
                    # print('f rank', f.rank)
                    # for u in f.subfaces:
                    #     print('u', u.color)
                    # print('f color', f.color)
                    assert(False)
                # In any case, insert g into Lₖ.
                L[k].append(g)
    if PROFILE:
        for k in range(0, d+1):
            print('L_%d' % k, len(L[k]))
    
    # ==========================================================================
    # Phase 3: Update all marked faces.
    # ==========================================================================
    if DEBUG:
        print('PHASE 3')
    if PROFILE:
        step_1_time = 0
        step_2_time = 0
        step_3_time = 0
        step_4_time = 0
        step_4_hit = 0
        step_4_total = 0
        step_5_time = 0
        step_5_hit = 0
        step_5_total = 0
        step_6_time = 0
        red_count = 0
    for k in range(0, d+1):
        f_k = len(L[k])
        for i in range(f_k):
            g = L[k][i]
            if g.color == COLOR_AH_PINK:
                g.color = COLOR_AH_GREY
                # Add to grey subfaces of superfaces
                for u in g.superfaces:
                    u._grey_subfaces.append(g)
            elif g.color == COLOR_AH_CRIMSON:
                g.color = COLOR_AH_BLACK
                # Add to black subfaces of superfaces
                for u in g.superfaces:
                    u._black_subfaces.append(g)
            elif g.color == COLOR_AH_RED:
                pt = (I.A @ g.int_pt - I.b).flatten().T
                g_sv = get_sign(pt, eps)

                if PROFILE:
                    red_count += 1
                    t_start = time()
                # Step 1. Create g_a = g ∩ h⁺ and g_b = g ∩ h⁻. Remove g from
                # 𝓐(H) and Lₖ and replace with g_a, g_b.
                g_a = Node(k)
                g_a.color = COLOR_AH_GREY
                g_a._sv_key = np.concatenate((g._sv_key, np.array([1], int)))
                g_a._sv_key = g_sv.copy()
                g_a._sv_key[-1] = 1
                g_b = Node(k)
                g_b.color = COLOR_AH_GREY
                g_b._sv_key = np.concatenate((g._sv_key, np.array([-1], int)))
                g_b._sv_key = g_sv.copy()
                g_b._sv_key[-1] = -1
                I.remove_node(g)
                L[k][i] = g_a
                L[k].append(g_b)
                I.add_node(g_a)
                I.add_node(g_b)

                if PROFILE:
                    step_1_time += time() - t_start
                    t_start = time()
                # Step 2. Create the black face f = g ∩ h, connect it to g_a and
                # g_b, and put f into 𝓐(H) and Lₖ₋₁.
                f = Node(k-1)
                f.color = COLOR_AH_BLACK
                # f._sv_key = np.concatenate((g._sv_key, np.array([0], int)))
                f._sv_key = g_sv.copy()
                f._sv_key[-1] = 0
                f.superfaces = [g_a, g_b]
                g_a.subfaces = [f]
                g_b.subfaces = [f]
                g_a._black_subfaces = [f]
                g_b._black_subfaces = [f]
                L[k-1].append(f)
                I.add_node(f)

                if PROFILE:
                    step_2_time += time() - t_start
                    t_start = time()
                # Step 3. Connect each red superface of g with g_a and g_b.
                for r in g.superfaces:
                    if DEBUG:
                        # if r.color != COLOR_AH_RED:
                        #     print('r', r.rank)
                        #     print('r', r._sv_key)
                        #     print('r', r._sv_key.astype(float))
                        #     # for u in 
                        assert(r.color == COLOR_AH_RED or r.rank == d+1)
                    # if r.color == COLOR_AH_RED:
                    g_a.superfaces.append(r)
                    g_b.superfaces.append(r)
                    r.subfaces.append(g_a)
                    r.subfaces.append(g_b)
                    r._grey_subfaces.append(g_a)
                    r._grey_subfaces.append(g_b)
                
                if PROFILE:
                    step_3_time += time() - t_start
                    t_start = time()
                # Step 4. Connect each white or grey subface of g with g_a if it
                # is in h⁺, and with g_b, otherwise.
                for u in g.subfaces:
                    if PROFILE:
                        step_4_total += 1
                    if u.color != COLOR_AH_WHITE and u.color != COLOR_AH_GREY:
                        if DEBUG:
                            assert(u.color == COLOR_AH_BLACK) # FIXME Can there be black subfaces?
                        continue
                    if u._sign_bit_n != n:
                        if PROFILE:
                            step_4_hit += 1
                        s = get_sign(a @ u.int_pt - b, eps)
                        u._sign_bit_n = n
                        u._sign_bit = s
                    else:
                        s = u._sign_bit
                    if s == 1:
                        g_a.subfaces.append(u)
                        if u.color == COLOR_AH_GREY:
                            g_a._grey_subfaces.append(u)
                        u.superfaces.append(g_a)
                    elif s == -1:
                        g_b.subfaces.append(u)
                        if u.color == COLOR_AH_GREY:
                            g_b._grey_subfaces.append(u)
                        u.superfaces.append(g_b)
                    else:
                        assert(False)
                
                if PROFILE:
                    step_4_time += time() - t_start
                    t_start = time()
                # Step 5. If k = 1, connect f with the -1 face, and connect f
                # with the black subfaces of the grey subfaces of g, otherwise.
                if k == 1:
                    zero = I.get(-1, 0)
                    f.subfaces.append(zero)
                    zero.superfaces.append(f)
                else:
                    # # VERSION 1
                    # V = dict()
                    # for u in g.subfaces:
                    #     if u.color != COLOR_AH_GREY:
                    #         continue
                    #     if DEBUG:
                    #         assert(u in g._grey_subfaces)
                    #     for v in u.subfaces:
                    #         if PROFILE:
                    #             step_5_total += 1
                    #         if v.color == COLOR_AH_BLACK:
                    #             if DEBUG:
                    #                 assert(v in u._black_subfaces)
                    #             V[tuple(v._sv_key)] = v
                    # V = list(V.values())

                    # # VERSION 2
                    # V = dict()
                    # for u in g._grey_subfaces:
                    #     for v in u._black_subfaces:
                    #         if PROFILE:
                    #             step_5_total += 1
                    #         V[tuple(v._sv_key)] = v
                    # V = list(V.values())
                    
                    # VERSION 3
                    V = list()
                    for u in g._grey_subfaces:
                        for v in u._black_subfaces:
                            if PROFILE:
                                step_5_total += 1
                            if v._black_bit == 0:
                                V.append(v)
                                v._black_bit = 1
                            else:
                                v._black_bit = 0

                    if PROFILE:
                        step_5_hit += len(V)
                    for v in V:
                        f.subfaces.append(v)
                        v.superfaces.append(f)
                
                if PROFILE:
                    step_5_time += time() - t_start
                    t_start = time()
                # Step 6. Update the interior points for f, g_a, and g_b.
                for u in [f, g_a, g_b]:
                    if u.rank == 0:
                        id0 = np.where(u._sv_key == 0)[0]
                        if DEBUG:
                            assert(len(id0) == d)
                        # TODO THIS IS OKAY I THINK
                        u.int_pt = np.linalg.solve(I.A[id0], I.b[id0])
                    elif u.rank == 1:
                        if len(u.subfaces) == 2:
                            p = np.array([v.int_pt.flatten() for v in u.subfaces]).T
                            u.int_pt = np.mean(p, axis=1, keepdims=True)
                        elif len(u.subfaces) == 1:
                            g_sv = u._sv_key[0:-1]
                            v = u.subfaces[0]
                            null = null_space(I.A[u._sv_key == 0], eps)
                            v_sv = get_sign(I.A @ v.int_pt - I.b, eps)
                            i = np.where(u._sv_key != v_sv)[0][0]
                            dot = I.A[i] @ null
                            if dot * u._sv_key[i] > 0:
                                u.int_pt = v.int_pt + null
                            else:
                                u.int_pt = v.int_pt - null
                        else:
                            assert(False)
                    else:
                        if DEBUG:
                            assert(len(u.subfaces) >= 2)
                        # p = np.array([v.int_pt.flatten() for v in u.subfaces[0:u.rank+1]]).T
                        p = np.array([v.int_pt.flatten() for v in u.subfaces]).T
                        u.int_pt = np.mean(p, axis=1, keepdims=True)
                    if DEBUG:
                        # Check interior point.
                        pt = (I.A @ u.int_pt - I.b).flatten().T
                        sv = get_sign(pt)
                        if not np.array_equal(u._sv_key, sv):
                            null = null_space(I.A[u._sv_key == 0], eps)
                            v_sv = get_sign(I.A @ v.int_pt - I.b, eps)

                            print('u', u._sv_key.astype(float))
                            print('v', v_sv.astype(float))
                            # i = np.where(u._sv_key != v_sv)[0].item()
                            inds = np.where(u._sv_key != v_sv)[0]
                            dot = I.A[i] @ null
                            print('dot', dot)

                            print('i', i)
                            print('rank e', np.linalg.matrix_rank(I.A[u._sv_key == 0], eps))
                            print('rank v', np.linalg.matrix_rank(I.A[v._sv_key == 0], eps))
                            print('null', null)
                            print('dot', dot)
                            print('u', u._sv_key.astype(float))
                            print('u rank', u.rank)
                            if u.rank >= 1:
                                for v in u.subfaces:
                                    print('v', v._sv_key.astype(float))
                            print('p', sv.astype(float))
                        assert(np.array_equal(u._sv_key, sv))
                        # Check for duplicates in super/sub faces.
                
                if PROFILE:
                    step_6_time += time() - t_start
                    t_start = time()
    if PROFILE:
        print(' # red: %d' % (red_count,))
        print('step 1: %0.8f' % (step_1_time))
        print('step 2: %0.8f' % (step_2_time))
        print('step 3: %0.8f' % (step_3_time))
        print('step 4: %0.8f %d/%d' % (step_4_time, step_4_hit, step_4_total))
        print('step 5: %0.8f %d/%d' % (step_5_time, step_5_hit, step_5_total))
        print('step 6: %0.8f' % (step_6_time))
        print('   k=0: %0.8f' % (0))
        print('   k=1: %0.8f' % (0))
        print('  k>=2: %0.8f' % (0))
    # Clear all colors, grey and black subface lists.
    for k in range(0, d+1):
        for f in L[k]:
            f.color = COLOR_AH_WHITE
            f._grey_subfaces.clear()
            f._black_subfaces.clear()

    return I