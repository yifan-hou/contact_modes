import numpy as np

from contact_modes.geometry import *


def build_normal_eqs_2d(manifolds):
    n_contacts = len(manifolds)
    n_dofs = len(manifolds[0].shape_A.get_dof_mask())
    A = np.zeros((n_contacts, n_dofs))
    B = np.array([0, 0, 1., 0, 0, 0]).reshape((6,1))
    k = 0
    for i in range(n_contacts):
        m = manifolds[i]
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
            A[k,None,:] -= J_h
            d_k += 1
        if d_k > 0:
            k += 1
    A = -A[0:k,:]
    b = np.array([[m.dist] for m in manifolds])
    return A, b

def build_tangent_eqs_2d(manifolds):
    pass

def enum_cs_modes_2d(manifolds):
    A, b = build_normal_eqs_2d(manifolds)

def enum_ss_modes_2d(manifolds, cs_mode):
    pass