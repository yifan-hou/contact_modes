import numpy as np

from .se3 import SE3
from .so3 import SO3

DEBUG = False

def build_normal_velocity_constraints(manifolds):
    # Create halfspace inequalities, Aq̇ - d ≤ 0.
    n_contacts = len(manifolds)
    n_dofs = len(manifolds[0].shape_A.get_dof_mask())
    A = np.zeros((n_contacts, n_dofs))
    if DEBUG:
        print('n_contacts', n_contacts)
        print('n_dofs', n_dofs)
    B = np.array([0, 0, 1., 0, 0, 0]).reshape((6,1))
    k = 0
    for i in range(n_contacts):
        m = manifolds[i]
        if DEBUG:
            print(m)
        body_A = m.shape_A
        body_B = m.shape_B
        d_k = 0
        if body_A.num_dofs() > 0:
            try:
                g_wo = body_A.get_transform_world()
                g_wc = m.frame_A()
                g_oc = SE3.inverse(g_wo) * g_wc
                J_b = body_A.get_body_jacobian()
                Ad_g_co = SE3.Ad(SE3.inverse(g_oc))
                J_h = B.T @ Ad_g_co @ J_b
                if DEBUG:
                    print('g_wo')
                    print(g_wo)
                    print('g_wc')
                    print(g_wc)
                    print('g_oc')
                    print(g_oc)
                    print('J_b')
                    print(J_b)
                    print(Ad_g_co)
                    print(B.T)
                    print(J_h)
                A[k,None,:] += J_h
                d_k += 1
            except:
                print(m)
                assert(False)
        if body_B.num_dofs() > 0:
            g_wo = body_B.get_transform_world()
            g_wc = m.frame_B()
            g_oc = SE3.inverse(g_wo) * g_wc
            J_s = body_B.get_spatial_jacobian()
            J_b = body_B.get_body_jacobian()
            Ad_g_co = SE3.Ad(SE3.inverse(g_oc))
            J_h = B.T @ Ad_g_co @ J_b
            if DEBUG:
                print('q')
                print(body_B.q)
                print('g_wo')
                print(g_wo)
                print('g_wc')
                print(g_wc)
                print('g_oc')
                print(g_oc)
                print('J_s')
                print(J_s)
                print('J_b')
                print(J_b)
                print(Ad_g_co)
                print(B.T)
                print(J_h)
            A[k,None,:] -= J_h
            d_k += 1
        if d_k > 0:
            k += 1
    A = -A[0:k,:]
    b = np.array([[m.dist] for m in manifolds])
    return A, b

def build_tangential_velocity_constraints(manifolds):
    # Create halfspace inequalities, Ax - b ≤ 0.
    pass