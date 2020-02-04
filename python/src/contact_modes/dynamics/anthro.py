import numpy as np

from contact_modes import SE3, SO3
from contact_modes.shape import Box

from .body import *
from .link import *
from .static import *
from .tree import *


class AnthroHand(Tree):
    def __init__(self):
        self.init()

    def init(self, num_fingers = 3, 
                   num_digits = 3,
                   finger_length = 5, 
                   finger_width = 2, 
                   finger_thickness = 0.6,
                   palm_length = 10.0, 
                   palm_width = 10.0,
                   palm_thickness = 0.6):
        # Create anthropomorphic hand.
        num_dofs = num_fingers * num_digits + (num_digits - 1)
        self.links = []
        # Create gₛₗ(0), ξ's, and mask for each finger.
        dof_id = 0
        for i in range(num_fingers + 1):
            finger = [Link('f%d_%d' % (i, 0))]
            # Finger digit 0 position relative to palm.
            g_sl0 = SE3.exp(np.array([(palm_width-finger_width)/(num_fingers-1)*i - (palm_width-finger_width)/2, 
                                      palm_length/2 + finger_length/2, 
                                      0, 0, 0, 0]))
            if i == num_fingers: # thumb
                g_sl0 = SE3.exp(np.array(
                    [(palm_width+finger_width)/2,
                    finger_length/2,
                    0, 0, 0, 0]))
            finger[0].set_transform_0(g_sl0)
            # Joint twist for digit 0.
            w_0 = np.array([1, 0, 0])
            p_0 = finger[0].get_transform_0().t.copy()
            p_0[1,0] -= finger_length/2
            xi_0 = np.zeros((6,1))
            xi_0[0:3,0,None] = -SO3.ad(w_0) @ p_0
            xi_0[3:6,0] =  w_0
            joint_twists = [xi_0]
            finger[0].set_joint_twists(joint_twists.copy())
            # Shape for digit 0.
            finger[0].set_shape(Box(finger_width, finger_length, finger_thickness))
            # Mask for digit 0.
            mask = np.array([False] * num_dofs)
            mask[dof_id] = True
            dof_id += 1
            finger[0].set_dof_mask(mask.copy())
            # Subsequent digit positions relative to previous digit.
            n_d = num_digits
            if i == num_fingers:
                n_d -= 1
            for j in range(1, n_d):
                # Link.
                digit = Link('f%d_%d' % (i, j))
                finger.append(digit)
                # Create gₛₗ(0).
                xi_0 = np.zeros((6,1))
                xi_0[0:3,0,None] = finger[j-1].get_transform_0().t
                xi_0[1,0] += finger_length
                g_slj = SE3.exp(xi_0)
                # Set gₛₗ(0).
                digit.set_transform_0(g_slj)
                # Add joint twist ξⱼ.
                w_j = np.array([1, 0, 0])
                p_j = g_slj.t.copy()
                p_j[1,0] -= finger_length/2
                xi_j = np.zeros((6,1))
                xi_j[0:3,0,None] = -SO3.ad(w_j) @ p_j
                xi_j[3:6,0] =  w_j
                joint_twists.append(xi_j)
                finger[j].set_joint_twists(joint_twists.copy())            
                # Shape for digit j.
                finger[j].set_shape(Box(finger_width, finger_length, finger_thickness))
                # Mask for digit j.
                mask[dof_id] = True
                dof_id += 1
                finger[j].set_dof_mask(mask.copy())
            # Add finger links.
            self.links.extend(finger)
        # Add palm.
        self.links.append(Static('palm'))
        self.links[-1].set_shape(Box(palm_width, palm_length, palm_thickness))
        self.links[-1].set_dof_mask(np.array([False] * num_dofs))
        # Set to 0 position.
        self.set_dofs(np.zeros((num_dofs,1)))
        # Create tree.
        self.init_tree()