# -*- coding: utf-8 -*-
import numpy as np

from contact_modes.exp import epsilon, sqrt_epsilon

from .jda import jda, zbitl
from .manifold import CollisionManifold


def cross(x, y):
    return np.cross(x.reshape((3,1)), y.reshape((3,1)), axis=0)

def create_manifold_gjk(d, p_A, p_B, obj_A, obj_B):
    manifold = CollisionManifold()
    manifold.pts = (p_A + p_B) / 2.0
    manifold.pts_A = p_A
    manifold.pts_B = p_B
    manifold.normal = np.sign(d) * (p_B - p_A) / np.linalg.norm(p_B - p_A)
    manifold.dist = d
    manifold.shape_A = obj_A
    manifold.shape_B = obj_B
    return manifold

def gjk(obj_A, obj_B, v=None):
    if v is None or np.linalg.norm(v) < sqrt_epsilon():
        v = np.ones((3,1))
    obj_A.supmap(-v)
    obj_B.supmap(v)
    v = obj_A.supmap(-v)-obj_B.supmap(v) # arbitrary point in A-B
    Y = np.zeros((3,4))
    A = np.zeros((3,4))
    B = np.zeros((3,4))
    b = 0
    d = np.zeros((4,4,3))
    d2 = np.zeros((4,4))
    dX = np.zeros((16,4))
    eps_abs = 10*epsilon()
    eps_rel = 100*sqrt_epsilon()
    lamb = np.array([[1.0],[0.0],[0.0],[0.0]])
    while True:
        A[:,zbitl(b),None] = obj_A.supmap(-v)
        B[:,zbitl(b),None] = obj_B.supmap(v)
        w = A[:,zbitl(b),None]-B[:,zbitl(b),None]
        # w ∈ Y
        cyclic = False
        for i in range(4):
            if b & (1 << i):
                if np.linalg.norm(Y[:,i,None]-w) < eps_abs:
                    cyclic = True
        vv = np.dot(v.T,v).item()
        vw = np.dot(v.T,w).item()
        # If w ∈ Y or |v|² - v⋅w ≤ ɛ²|v|², then the objects intersect only in
        # the margins or not at all.
        if cyclic or (vv - vw <= eps_rel**2*vv):
            N = v / np.sqrt(vv)
            u_A = obj_A.margin()
            u_B = obj_B.margin()
            p_A = np.matmul(A, lamb)
            p_B = np.matmul(B, lamb)
            d = np.sqrt(vv) - u_A - u_B
            p_A = p_A - u_A*N
            p_B = p_B + u_B*N
            return create_manifold_gjk(d, p_A, p_B, obj_A, obj_B)
        Y[:,zbitl(b),None] = w
        # ν = v(conv(Y)) but also ν = Yλ
        v, lamb, b, d, d2, dX = jda(Y, b, d, d2, dX)
        # max |y|², y ∈ W
        y_max = 0
        for i in range(4):
            if b & (1 << i):
                y_sqr = np.dot(Y[:,i],Y[:,i])
                if y_max < y_sqr:
                    y_max = y_sqr
        vv = np.dot(v.T,v).item()
        # If |W| == 4 or |v|² ≤ ɛ|y|², then the original objects intersect as
        # well.
        if b == 15 or vv <= eps_abs * y_max:
            d = np.linalg.norm(v)
            p_A = np.matmul(A, lamb)
            p_B = np.matmul(B, lamb)
            return create_manifold_gjk(d, p_A, p_B, obj_A, obj_B)
