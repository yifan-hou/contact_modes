import numpy as np


def intersect(F, G):
    H = Face(tuple(set(F.verts) & set(G.verts)), F.dim()-1)
    H.parents = [F, G]
    if len(H.verts) < F.d:
        return None
    if H.verts:
        return H
    else:
        return None

class Face(object):
    def __init__(self, verts, d):
        self.verts = verts
        self.d = d
        self.parents = None

    def dim(self):
        # Return dim(aff(F))
        return self.d

class Facet(Face):
    def __init__(self):
        pass

class Vertex(Face):
    def __init__(self):
        pass

class FaceLattice(object):
    def __init__(self, M=None, d=None):
        if M is not None:
            self.build(M, d)

    def euler_characteristic(self):
        pass

    def build(self, M, d):
        assert(M is not None)
        assert(d is not None)

        # Rows are vertices, columns are faces.
        n_verts = M.shape[0]
        n_facets = M.shape[1]

        # Create graded lattice.
        L = []

        # Create polytope face.
        P = Face(tuple(range(n_verts)), d)
        L.append([P])

        # Create facets.
        # print('rank', d-1)
        L.append([])
        for i in range(n_facets):
            F = Face(tuple(np.where(M[:,i] > 0)[0]), d-1)
            F.parents = [P]
            # print(F.verts)
            L[-1].append(F)

        # Create faces.
        for di in range(d-1):
            # print('rank', d-2-di)
            H = L[-1]
            L.append([])
            vert_sets = set()
            for i in range(len(H)):
                for j in range(i + 1, len(H)):
                    if i == j:
                        continue
                    J = intersect(H[i], H[j])
                    if J is None:
                        continue
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

    def mode_strings(self):
        modes = []
        L = self.L
        n_verts = len(L[0][0].verts)
        for i in range(len(L)):
            for j in range(len(L[i])):
                m = np.array(['s'] * n_verts)
                # print(L[i][j].verts)
                m[list(L[i][j].verts)] = 'c'
                modes.append(m.tolist())
        return np.array(sorted(modes))
        
    def hasse_diagram(self, dot_file):
        pass