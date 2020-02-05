from time import time

import numpy as np

DEBUG = False

def intersect(F, G):
    H = Face(tuple(set(F.verts) & set(G.verts)), F.dim()-1)
    H.parents = [F, G]
    if len(H.verts) < F.d:
        return None
    if H.verts:
        return H
    else:
        return None

def merge_sorted(a, b):
    pass

def difference_sorted(a, b):
    # Return a / b
    n = len(a)
    m = len(b)
    i = 0
    j = 0
    c = []
    while i < n:
        if j == m:
            c.append(a[i])
            i += 1
        elif a[i] < b[j]:
            c.append(a[i])
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            i += 1
            j += 1
    return tuple(c)

def intersect_sorted(a, b):
    n = len(a)
    m = len(b)
    i = 0
    j = 0
    c = []
    while i < n and j < m:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            c.append(a[i])
            i += 1
            j += 1
    # c = np.array(c)

    # if DEBUG:
    #     assert(np.all(c == np.array(sorted(list(set(a) & set(b))))))
    
    return tuple(c)

def intersect_preimages(A, B):
    if len(A) == 0:
        return ()
    I = B[A[0]]
    for i in range(1, len(A)):
        I = intersect_sorted(I, B[A[i]])
    return I

def closure(S, V, F):
    return intersect_preimages(intersect_preimages(S, F), V)

class Face(object):
    def __init__(self, verts, d, v=None, f=None):
        self.verts = verts
        self.d = d
        self.parents = []
        self.children = []
        self.v = v
        self.f = f
        if v is None:
            self.v = []
        if f is None:
            self.f = ()

    def dim(self):
        # Return dim(aff(F))
        return self.d

class FaceLattice(object):
    def __init__(self, M=None, d=None):
        if M is not None:
            # t_start = time()
            self.build(M, d)
            # print('build', time()-t_start)

            # t_start = time()
            # self.build_fast(M, d)
            # print('build fast', time()-t_start)

    def euler_characteristic(self):
        pass

    def build_fast(self, M, d):
        # If needed, convert to polar vertex-facet incidence matrix so that
        # algorithm runs in O(min{m,n}⋅α⋅φ) time.
        polar = False
        # if M.shape[1] < M.shape[0]:
        #     M = M.T
        #     polar = True

        # Rows are vertices, columns are faces.
        n_verts = M.shape[0]
        n_facets = M.shape[1]

        # Create sorted sparse format of M.
        F = []
        V = []
        for i in range(n_verts):
            F.append(tuple(np.where(M[i,:])[0]))
        for i in range(n_facets):
            V.append(tuple(np.where(M[:,i])[0]))

        # Create lattice.
        L = [dict() for i in range(d+2)]
        L[0][()] = Face((), 
                        0, 
                        v=[len(F[i]) for i in range(n_verts)], 
                        f=(range(n_facets)))

        # Main loop.
        verts = tuple(range(n_verts))
        for i in range(d+1):
            Q = L[i].values()
            for H in Q:
                V_H = difference_sorted(verts, H.verts)
                color = [0] * n_verts
                for v in V_H:
                    # Closure of H + v in O(n) time, hopefully.
                    if False:
                        G_f = intersect_sorted(H.f, F[v])
                        G_v = H.v.copy()
                        for w in difference_sorted(H.f, G_f):
                            for u in V[w]:
                                G_v[u] -= 1
                        G_v_max = np.amax(G_v)
                        if G_v_max == 0:
                            continue
                        G = tuple(np.argwhere(G_v == G_v_max).flatten().tolist())
                    else:
                        G_v = []
                        G_f = ()
                        G = closure(H.verts + (v,), V, F)
                    if DEBUG:
                        G0 = closure(H.verts + (v,), V, F)
                        print('G', G)
                        print('G0', G0)
                        assert(G0 == G)
                    if len(G) == 0:
                        continue
                    color[v] = 1
                    W = difference_sorted(G, H.verts)
                    for w in W:
                        if w == v:
                            continue
                        if color[w] >= 0:
                            color[v] = -1
                            break
                    if color[v] == 1:
                        if not L[i+1].get(G):
                            L[i+1][G] = Face(G, H.d + 1, G_v, G_f)
                        H.parents.append(L[i+1][G])
                        L[i+1][G].children.append(H)
        
        # Add arcs to P.
        L[d+1][0] = Face(tuple(range(n_verts)), d + 1)
        Q = L[d].values()
        for H in Q:
            H.parents.append(L[d+1][0])
            L[d+1][0].children.append(H)
        
        # Flip lattice to match our convention.
        self.L = [[] for i in range(d+2)]
        for i in range(d+2):
            self.L[i] = list(L[d+1-i].values())

    def build(self, M, d):
        # Rows are vertices, columns are faces.
        n_verts = M.shape[0]
        n_facets = M.shape[1]
        print('# verts', n_verts)
        print('# facets', n_facets)

        # Initialize lattice.
        L = []

        # Create d face.
        P = Face(tuple(range(n_verts)), d)
        L.append([P])

        # Create d-1 faces (facets).
        L.append([])
        for i in range(n_facets):
            F = Face(tuple(np.where(M[:,i] > 0)[0]), d-1)
            F.parents = [P]
            L[-1].append(F)

        # Create k faces, k ≥ 0.
        for k in range(d-2, -1, -1):
            H = L[-1]
            num_super_faces = len(H)
            L.append([])
            vert_sets = set()
            faces = dict()
            for i in range(num_super_faces):
                for j in range(i + 1, num_super_faces):
                    if i == j:
                        continue
                    v = intersect_sorted(H[i].verts, H[j].verts)
                    J = intersect(H[i], H[j])

                    if J is None:
                        continue

                    J.verts = tuple(sorted(J.verts))
                    assert(v == J.verts)
                    
                    # if not v:
                    #     continue
                    # if len(v) < k + 1:
                    #     continue

                    # print('  v:', v)
                    # print('J.v:', J.verts)

                    # if v in faces.keys():
                    #     if H[i] not in faces[v].parents:
                    #         faces[v].parents.append(H[i])
                    #     if H[j] not in faces[v].parents:
                    #         faces[v].parents.append(H[j])
                    if J.verts in faces.keys():
                        if H[i] not in faces[J.verts].parents:
                            faces[J.verts].parents.append(H[i])
                        if H[j] not in faces[J.verts].parents:
                            faces[J.verts].parents.append(H[j])
                    else:
                        faces[J.verts] = J
                    if J.verts in vert_sets:
                        continue
                    vert_sets.add(J.verts)
                    L[-1].append(J)
                    # print(J.verts)

        # Create empty face.
        E = Face([], 0)
        E.parents = L[-1]
        L.append([E])

        # Store face lattice.
        self.L = L

    def append_empty(self):
        # Create empty face.
        E = Face([], 0)
        E.parents = None
        self.L.append([E])

    def rank(self):
        return len(self.L)-1

    def num_proper_faces(self):
        cnt = 0
        L = self.L
        for i in range(1, len(L)-1):
            cnt += len(L[i])
        return cnt
    
    def num_faces(self):
        cnt = 0
        L = self.L
        for i in range(len(L)):
            cnt += len(L[i])
        return cnt

    def num_k_faces(self, k):
        if k == -1 and len(self.L) == 1:
            return 0
        r = len(self.L)-2
        return len(self.L[r-k])
    
    def csmodes(self, mask, dual_map):
        cs_modes = []
        L = self.L
        n_pts = len(L[0][0].verts)
        if mask is not None:
            n_pts = len(mask)
        n_nomask = n_pts - np.sum(mask)
        n_verts = len(L[0][0].verts)
        for i in range(len(L)):
            for j in range(len(L[i])):
                F = list(L[i][j].verts)
                v_mode = np.array(['s']*n_verts)
                v_mode[F] = 'c'
                d_mode = np.array(['s']*n_nomask)
                for k in range(n_verts):
                    d = dual_map[k]
                    for l in d:
                        d_mode[l] = v_mode[k]
                cs_mode = np.array(['c']*n_pts)
                if mask is not None:
                    cs_mode[~mask] = d_mode
                else:
                    cs_mode = d_mode
                cs_modes.append(cs_mode.tolist())
                L[i][j].m = cs_mode
        return np.array(cs_modes)
        
    def hesse_diagram(self, dot_file):
        with open(dot_file, 'w') as dot:
            G = []
            G.append('graph {')
            G.append('node [shape=circle, label="", style=filled, color="0 0 0", width=0.25]')
            G.append('splines=false')
            G.append('layout=neato')
            # G.append('P')
            # G.append('E')

            # Create ranks manually
            rank_sep = 0.9
            node_sep = 0.50
            names = dict()
            L = self.L
            for i in range(len(L)):
                n_f = len(L[i])
                l = (n_f - 1) * node_sep
                for j in range(len(L[i])):
                    F = L[i][j]
                    f_n = 'f%d_%d' % (i,j)
                    names[F] = f_n
                    x = -l/2 + j * node_sep
                    y = -rank_sep * i
                    G.append(f_n + ' [pos="%f,%f!"]' % (x,y))
            
            # Create lattice
            for i in range(len(L)):
                for j in range(len(L[i])):
                    F = L[i][j]
                    f_n = names[F]
                    print(f_n)
                    if F.parents is None:
                        continue
                    for H in F.parents:
                        h_n = names[H]
                        G.append(f_n + '--' + h_n)

            G.append('}')

            dot.write('\n'.join(G) + '\n')