import numpy as np
from scipy.linalg import null_space

from contact_modes import lexographic_combinations

from .incidence_graph import *


DEBUG=True

COLOR_WHITE=0
COLOR_PINK =1
COLOR_RED =2
COLOR_CRIMSON =3
COLOR_GREY =4
COLOR_BLACK =5
COLOR_GREEN =6

import operator as op
from functools import reduce

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
        return COLOR_CRIMSON
    else:
        return COLOR_WHITE

def color_edge(e, a, b, eps=np.finfo(np.float32).eps):
    # Assume |a| = 1.
    if len(e.subfaces) == 2:
        v0 = e.subfaces[0]
        v1 = e.subfaces[1]
        s0 = sign_vertex(v0, a, b, eps)
        s1 = sign_vertex(v1, a, b, eps)
        if s0 * s1 == 1:
            return COLOR_WHITE
        elif s0 == 0 and s1 == 0:
            return COLOR_CRIMSON
        elif s0 == 0 or s1 == 0:
            return COLOR_PINK
        elif s0 + s1 == 0:
            return COLOR_RED
        else:
            assert(False)
    elif len(e.subfaces) == 1:
        v0 = e.subfaces[0]
        s0 = sign_vertex(v0, a, b, eps)
        v_e = vector(e)
        s_e = get_sign(a@v_e, eps).item()
        if s0 == 0 and s_e == 0:
            return COLOR_CRIMSON
        elif s0 == 0 and s_e != 0:
            return COLOR_PINK
        elif s0 * s_e == 1:
            return COLOR_WHITE
        elif s0 * s_e == -1:
            return COLOR_RED
        elif s0 != 0 and s_e == 0:
            return COLOR_WHITE
        # TODO Check above conditions.
        else:
            assert(False)
    else:
        assert(False)

def initial_arrangement(A, b, eps=np.finfo(np.float32).eps):
    # Assert we are given d linearly independent hyperplanes.
    n = A.shape[0]
    d = A.shape[1]
    assert(n == d)
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
            f.sign_vector = np.ones((d,), int)
            f.sign_vector[c] = -1
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
                sign_vector = g.sign_vector.copy()
                if sign_vector[i] == 0:
                    continue
                sign_vector[i] = 0
                f = I.find(sv=sign_vector, r=k)
                if f is None:
                    f = Node(k)
                    f.sign_vector = sign_vector
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
        null = null_space(A[f.sign_vector == 0])
        dot = A[f.sign_vector != 0] @ null
        i = np.where(f.sign_vector != 0)[0][0]
        if dot * f.sign_vector[i] > 0:
            f.int_pt = vertex.int_pt + null
        else:
            f.int_pt = vertex.int_pt - null
        if DEBUG:
            # print(' f.sv', f.sign_vector)
            pt = (A @ f.int_pt - b).flatten().T
            sv = get_sign(pt)
            assert(np.array_equal(f.sign_vector, sv))
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
                assert(np.array_equal(f.sign_vector, sv))
                # print(' f.sv', f.sign_vector)
                # print('A@f-b', (A @ f.int_pt - b).flatten().T)

    return I

def increment_arrangement(a, b, I, eps=np.finfo(np.float32).eps):
    # Normalize halfspace, |a| = 1.
    norm_a = np.linalg.norm(a)
    a = a / norm_a
    b = b / norm_a

    # ==========================================================================
    # Phase 1: Find an edge e₀ in 𝓐(H) such that cl(e₀) ∩ h ≠ ∅
    # ==========================================================================
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
        if color_edge(e0, a, b, eps) > COLOR_WHITE:
            if DEBUG:
                print(e0.sign_vector)
                print('color', color_edge(e0, a, b, eps))
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
        min_dist = np.inf
        for e in v.superfaces:
            if e is e0:
                continue
            v_e = vector(e) / np.linalg.norm(vector(e))
            dist = np.linalg.norm(v_e - (v_e0.T @ v_e) * v_e0)
            if dist < min_dist:
                e_min = e
        e0 = e_min
    
    # ==========================================================================
    # Phase 2: Mark all faces f with cl(f) ∩ h ≠ ∅ pink, red, or crimson.
    # ==========================================================================
    # Add some 2 face incident upon e₀ to Q and mark it green.
    f = e0.superfaces[0]
    f.color = COLOR_GREEN
    Q = [f]
    # Color vertices, edges, and 2 faces of 𝓐(H).
    d = a.shape[1]
    L = [[] for i in range(d+1)]
    while Q:
        f = Q.pop(0)
        for e in f.subfaces:
            if e.color != COLOR_WHITE:
                continue
            color_e = color_edge(e, a, b, eps)
            if color_e > COLOR_WHITE:
                # Mark each white vertex v ∈ h crimson and insert v into L₀.
                for v in e.subfaces:
                    if v.color == COLOR_WHITE:
                        color_v = color_vertex(v, a, b, eps)
                        if color_v == COLOR_CRIMSON:
                            v.color = color_v
                            L[0].append(v)
                # Color e and insert e into L₁.
                e.color = color_e
                L[1].append(e)
                # Mark all white 2 faces green and put them into Q.
                for g in e.superfaces:
                    if g.color == COLOR_WHITE:
                        g.color = COLOR_GREEN
                        Q.append(g)
    # Color k faces, 2 ≤ k ≤ d.
    for k in range(2, d+1):
        for f in L[k-1]:
            for g in f.superfaces:
                if g.color != COLOR_WHITE and g.color != COLOR_GREEN:
                    continue
                if f.color == COLOR_PINK:
                    above = 0
                    below = 0
                    for f_g in g.subfaces:
                        if f_g.color == COLOR_RED:
                            above = 1
                            below = 1
                            break
                        s = get_sign(f_g.int_pt.T @ a - b, eps)
                        if s > 0:
                            above = 1
                        elif s < 0:
                            below = 1
                    if above * below == 1:
                        g.color = COLOR_RED
                elif f.color == COLOR_RED:
                    g.color = COLOR_RED
                elif f.color == COLOR_CRIMSON:
                    crimson = True
                    for f_g in g.subfaces:
                        if f_g.color != COLOR_CRIMSON:
                            crimson = False
                            break
                    if crimson:
                        g.color = COLOR_CRIMSON
                    else:
                        g.color = COLOR_PINK
                else:
                    assert(False)
                # In any case, insert g into Lₖ.
                L[k].append(g)
    
    # ==========================================================================
    # Phase 3: Update all marked faces.
    # ==========================================================================
    for k in range(0, d+1):
        for i in range(len(L[k])):
            g = L[k][i]
            if g.color == COLOR_PINK:
                g.color = COLOR_GREY
            elif g.color == COLOR_CRIMSON:
                g.color = COLOR_BLACK
            elif g.color == COLOR_RED:
                pass