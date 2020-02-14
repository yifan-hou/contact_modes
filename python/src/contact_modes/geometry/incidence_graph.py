import numpy as np

DEBUG=False

COLOR_AH_WHITE=0
COLOR_AH_PINK =1
COLOR_AH_RED =2
COLOR_AH_CRIMSON =3
COLOR_AH_GREY =4
COLOR_AH_BLACK =5
COLOR_AH_GREEN =6

def lexsort(S):
    idx = np.lexsort(np.rot90(S))
    return S[idx]

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

def build_vertex_facet_matrix(ret, verts):
    # Build facet-vertex incidence matrix.
    n_facets = int(ret[0])
    M = np.zeros((len(verts), n_facets), int)
    for i in range(1, len(ret)):
        vert_set = [int(x) for x in ret[i].split(' ')][1:]
        for v in vert_set:
            M[v,i-1] = 1
    if DEBUG:
        print('M')
        print(M)
    return M

def build_incidence_graph(M, d):
    """Build incidence graph from vertex-facet incidence matrix M.
    
    Arguments:
        M {|v|x|f| matrix} -- Vertex-facet incidence matrix
        d {int} -- Dimension
    
    Returns:
        IncidenceGraph -- The resulting face lattice.
    """
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

    # Create incidence graph.
    I = IncidenceGraph(d-1)
    zero = Node(-1)
    zero._sv_key = ()
    I.add_node(zero)

    # Main loop.
    verts = tuple(range(n_verts))
    for k in range(-1, d):
        Q = I.rank(k).values()
        for H in Q:
            V_H = difference_sorted(verts, H._sv_key)
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
                    G = closure(H._sv_key + (v,), V, F)
                if DEBUG:
                    G0 = closure(H._sv_key + (v,), V, F)
                    print('G', G)
                    print('G0', G0)
                    assert(G0 == G)
                if len(G) == 0:
                    continue
                color[v] = 1
                W = difference_sorted(G, H._sv_key)
                for w in W:
                    if w == v:
                        continue
                    if color[w] >= 0:
                        color[v] = -1
                        break
                if color[v] == 1:
                    if not I.rank(k+1).get(G):
                        if DEBUG:
                            assert(H.d + 1 == k)
                        u = Node(H.rank+1)
                        u._sv_key = G
                        I.add_node(u)
                        # I.rank(k+1)[G] = Face(G, H.d + 1, G_v, G_f)
                    H.superfaces.append(I.rank(k+1)[G])
                    I.rank(k+1)[G].subfaces.append(H)

    return I

class Node(object):
    def __init__(self, k):
        self.color = 0
        self.rank = k
        self.pos = None
        self.int_pt = None
        self.superfaces = []
        self.subfaces = []
        self._sv_key = () # FIXME NEVER USE THIS ITS ACTUALLY AN INDEX
        self._grey_subfaces = []
        self._black_subfaces = []
        self._black_bit = 0

class IncidenceGraph(object):
    def __init__(self, d):
        self.lattice = [dict() for i in range(d + 3)]
        self.A = np.zeros((0,d))
        self.b = np.zeros((0,1))

    def dim(self):
        return self.A.shape[1]

    def halfspaces(self):
        return self.A, self.b

    def num_halfspaces(self):
        return self.A.shape[0]

    def num_incidences(self):
        f_sum = 0
        for i in range(self.dim() + 1):
            for f in self.rank(i).values():
                f_sum += len(f.superfaces) + len(f.subfaces)
        return f_sum
    
    def num_k_faces(self, k):
        return len(self.rank(k))

    def add_halfspace(self, a, b):
        self.A = np.concatenate((self.A, a), axis=0)
        self.b = np.concatenate((self.b, b), axis=0)

    def add_node(self, node):
        self.rank(node.rank)[tuple(node._sv_key)] = node

    def remove_node(self, node):
        # Remove arcs.
        for f in node.subfaces:
            f.superfaces.remove(node)
        for g in node.superfaces:
            g.subfaces.remove(node)
            if node.color == COLOR_AH_GREY:
                g._grey_subfaces.remove(node)
            elif node.color == COLOR_AH_BLACK:
                g._black_subfaces.remove(node)
        # Remove node.
        del self.rank(node.rank)[tuple(node._sv_key)]

    def rank(self, k):
        return self.lattice[k+1]

    def find(self, sv=None, r=None):
        if sv is not None:
            sv = tuple(sv)
            if sv in self.rank(r):
                return self.rank(r)[sv]
            else:
                return None

    def get(self, r, i):
        return list(self.rank(r).values())[i]

    def get_positions(self, k):
        pos = []
        for u in self.rank(k).values():
            pos.append(u.pos)
        return np.array(pos)

    def sign_vectors(self, k, I=None, eps=np.finfo(np.float32).eps):
        sv = []
        for u in self.rank(k).values():
            s = self.get_sign(self.A @ u.int_pt - self.b, eps).tolist()
            if I is not None:
                s = [s[i] for i in I]
            sv.append(s)
        return np.array(sv)
    
    def get_sign(self, v, eps=np.finfo(np.float32).eps):
        v = v.flatten()
        return np.asarray(np.sign(np.where(np.abs(v) > eps, v, np.zeros((v.shape[0],)))), int)