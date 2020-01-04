import numpy as np
import scipy as sp
import pyhull
from pyhull.halfspace import Halfspace
from .helpers import hat, lexographic_combinations
from .interior_point import interior_point_halfspace


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
            b = np.zeros((H.shape[0], 1))
            x = interior_point_halfspace(H, b)

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

    # Create halfspace inequalities, Ax - b <= 0.
    n_pts = points.shape[1]
    A = np.zeros((n_pts, 6))
    for i in range(n_pts):
        A[i,0:3] = normals[:,i].flatten()
        A[i,3:6] = np.dot(normals[:,i].T, hat(points[:,i])).flatten()
    A *= -1
    b = np.zeros((n_pts, 1))
    print('A')
    print(A)

    # Get interior point using linear programming.
    int_pt = interior_point_halfspace(A, b) # Wow such math
    print('int_pt')
    print(int_pt)

    # Compute dual points.
    b_off = b - np.dot(A, int_pt)
    dual = A / b_off
    # dual = np.concatenate((dual, np.zeros((1,6))), axis=0)
    if DEBUG:
        print('b off')
        print(b_off)
        print('dual')
        print(dual)
    # dual = [list(dual[i,:]) for i in range(n_pts)]

    # Reduce dual points to minimally affinely independent set (?)
    null = sp.linalg.null_space((dual - dual[1,:]))
    orth = sp.linalg.orth((dual - dual[1,:]).T)
    if orth.shape[1] != 6:
        dual = np.dot((dual-dual[1,:]), orth)
        print('orth @ dual')
        print(dual)
    if DEBUG:
        print('null')
        print(null)
        print('orth')
        print(orth)

    # Compute dual convex hull.
    dual = [list(dual[i,:]) for i in range(n_pts)]
    print('dual')
    print(np.array(dual))
    ret = pyhull.qconvex('s Fn', dual)
    print(np.array(ret))

    return ret

    # Call qhalf.
    # Ab = np.concatenate((A, b), axis=1)
    # hs = [Halfspace(A[i,:], b[i,0]) for i in range(n_pts)]
    # print(pyhull.qhalf('s i', hs, int_pt))


def enum_sliding_sticking_3d(points, normals, cs_mode):
    pass