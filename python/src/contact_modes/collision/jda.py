# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import numpy.linalg as linalg
from numpy import dot, matmul
from numpy.linalg import det

from contact_modes import epsilon


def zbitl(x):
    # Find the right-most 0-bit.
    for i in range(4):
        if x & (1 << i) == 0:
            return i
    assert(i < 4)

def count(x):
    counts = bytearray(b'\x00\x01\x01\x02\x01\x02\x02\x03\x01\x02\x02\x03\x02\x03\x03\x04\x01\x02\x02\x03\x02\x03\x03\x04\x02\x03\x03\x04\x03\x04\x04\x05\x01\x02\x02\x03\x02\x03\x03\x04\x02\x03\x03\x04\x03\x04\x04\x05\x02\x03\x03\x04\x03\x04\x04\x05\x03\x04\x04\x05\x04\x05\x05\x06\x01\x02\x02\x03\x02\x03\x03\x04\x02\x03\x03\x04\x03\x04\x04\x05\x02\x03\x03\x04\x03\x04\x04\x05\x03\x04\x04\x05\x04\x05\x05\x06\x02\x03\x03\x04\x03\x04\x04\x05\x03\x04\x04\x05\x04\x05\x05\x06\x03\x04\x04\x05\x04\x05\x05\x06\x04\x05\x05\x06\x05\x06\x06\x07\x01\x02\x02\x03\x02\x03\x03\x04\x02\x03\x03\x04\x03\x04\x04\x05\x02\x03\x03\x04\x03\x04\x04\x05\x03\x04\x04\x05\x04\x05\x05\x06\x02\x03\x03\x04\x03\x04\x04\x05\x03\x04\x04\x05\x04\x05\x05\x06\x03\x04\x04\x05\x04\x05\x05\x06\x04\x05\x05\x06\x05\x06\x06\x07\x02\x03\x03\x04\x03\x04\x04\x05\x03\x04\x04\x05\x04\x05\x05\x06\x03\x04\x04\x05\x04\x05\x05\x06\x04\x05\x05\x06\x05\x06\x06\x07\x03\x04\x04\x05\x04\x05\x05\x06\x04\x05\x05\x06\x05\x06\x06\x07\x04\x05\x05\x06\x05\x06\x06\x07\x05\x06\x06\x07\x06\x07\x07\x08')
    return counts[x]

def vconv(Y, b):
    pass

def vaff(Y, b):
    # X
    m = count(b)
    assert(m > 0)
    assert(m <= 4)
    X = np.zeros((3,m))
    k = 0
    for i in range(4):
        if b & (1 << i):
            X[:,k] = Y[:,i]
            k += 1
    # A, b
    A = np.ones((m,m))
    for i in range(1,m):
        for j in range(m):
            A[i,j] = np.dot(X[:,i]-X[:,0], X[:,j])
    q = np.zeros((m,1))
    q[0] = 1
    assert(det(A) > 0) # true when X is affinely independent
    # λ
    l = linalg.solve(A, q)
    k = 0
    lamb = np.zeros((4,1))
    for i in range(4):
        if b & (1 << i):
            lamb[i,0] = l[k]
            k += 1
    # v
    v = matmul(Y, lamb)
    # Δ^X_i, i ∈ X
    dXi = lamb / det(A)
    return v, lamb, dXi

def deltaXi(Y, b, index):
    # Calculate Δ^X_i = (-1)^(1+j)det(A_1j), where i, j ∈ [1, m]
    # X
    assert(b & (1 << index))
    m = count(b)
    assert(m > 0)
    assert(m <= 4)
    X = np.zeros((3,m))
    k = 0
    j_ = 0
    for i in range(4):
        if b & (1 << i):
            if i == index:
                j_ = k
            X[:,k] = Y[:,i]
            k += 1
    assert(0 <= j_)
    assert(j_ < m)
    # A
    A = np.ones((m,m))
    for i in range(1,m):
        for j in range(m):
            A[i,j] = np.dot(X[:,i]-X[:,0], X[:,j])
    # A_1,j
    A_1  = np.delete(A,0,axis=0)
    A_1j = np.delete(A_1,j_,axis=1)
    # Δ^X_i
    if m == 1:
        return 1
    else:
        # index from i, j ∈ [1 m] during cofactor expansion!
        return (-1)**(1+j_+1)*det(A_1j)

def jda1(Y, b):
    # Slower version of Johnson's distance algorithm where Δ^X_i is recomputed
    # for every subset X∪{w_k} ∈ Y.
    w = 1 << zbitl(b)
    for x in range(1, b+w+1):
        if ((b+w) & x) != x:
            continue
        allpos = True
        offneg = True
        dX = np.zeros((4,1))
        for i in range(4):
            if x & (1 << i):
                dX[i,0] = deltaXi(Y,x,i)
                if dX[i,0] < 0:
                    allpos = False
            elif (b+w) & (1 << i):
                dXj = deltaXi(Y,x+(1<<i),i)
                if dXj >= 0:
                    offneg = False
        if allpos & offneg:
            break
    lamb = dX/np.sum(dX)
    v = matmul(Y,lamb)

    return v, lamb, x

def jda(Y, b, d, d2, dX):
    # Faster version of Johnson's distance algorithm which caches intermediate
    # variables.
    k = zbitl(b)
    for i in range(4):
        d_ik = Y[:,i]-Y[:,k]
        d[i,k,:] =  d_ik
        d[k,i,:] = -d_ik
        d2[i,k] = matmul(d_ik.transpose(), d_ik)
        d2[k,i] = d2[i,k]
    w = 1 << k

    eps = epsilon()
    for x in range(1,b+w+1):
        if ((b+w) & x) != x:
            continue
        if not (x & w): # only valid when Y[:,k] = supmap(-v)
            continue
        allpos = True
        offneg = True
        pc = count(x)
        for i in range(4):
            j = 1 << i
            if x & (1 << i):
                if pc == 1:
                    dX[x,i] = 1
                else:
                    k_min = 0
                    d2_min = np.inf
                    for k in range(4):
                        if x & (1 << k) & ~j:
                            if d2[k,i] < d2_min:
                                k_min = k
                                d2_min = d2[k,i]
                    k = k_min
                    dX[x,i] = matmul(d[k,i,None,:], matmul(Y, dX[x-j,:,None]))
                if dX[x,i] < 0:
                    allpos = False
                # dXi = deltaXi(Y,x,i)
                # assert(abs(dX[x,i]-dXi) < 1e-7)
        for i in range(4):
            j = 1 << i
            if (~x & j) & ((b+w) & j):
                # Search for k = argmin |d(k,i)|, k ∈ X
                k_min = 0
                d2_min = np.inf
                for k in range(4):
                    if x & (1 << k):
                        if d2[k,i] < d2_min:
                            k_min = k
                            d2_min = d2[k,i]
                k = k_min
                dX[x+j,i] = matmul(d[k,i,None,:], matmul(Y, dX[x,:,None]))
                if dX[x+j,i] >= 10*eps:
                    offneg = False
                # dXj = deltaXi(Y,x+j,i)
                # assert(abs(dX[x+j,i]-dXj) < 1e-7)
        if allpos & offneg:
            break
    # assert(allpos & offneg)
    
    lamb = dX[x,:,None]/np.sum(dX[x,:])
    v = matmul(Y,lamb)

    return v, lamb, x, d, d2, dX
