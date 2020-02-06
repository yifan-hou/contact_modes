import numpy as np

from contact_modes import SE3, SO3
from contact_modes.viewer.backend import *

from .icosphere import Icosphere

DEBUG = False

class Ellipse(Icosphere):
    def __init__(self, r1, r2, r3):
        super(Ellipse, self).__init__(refine=2)
        self.r = [r1, r2, r3]

    def draw(self, shader):
        shader.use()

        # Compute scaled transforms for the arrow.
        g = self.get_tf_world().matrix()
        s = np.diag([self.r[0], self.r[1], self.r[2], 1.0])
        shader.set_mat4('model', (g @ s).T)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_elems_draw)
        glBindVertexArray(0)

    def supmap(self, v):
        m = self.margin()
        r0 = self.r[0]-m
        r1 = self.r[1]-m
        r2 = self.r[2]-m

        tf = self.get_tf_world()

        # f_aff(v) = Bv + c
        B = tf.R.matrix() @ np.diag([r0, r1, r2])
        c = tf.t.copy()

        # v <- Bᵀv
        v = B.T @ v

        if DEBUG:
            print('v', v.T)
            print('|v|', np.linalg.norm(v))

        # supmap_sphere(v)
        if np.linalg.norm(v) < 1e-14:
            x = np.array([0., 0., 1.]).reshape((3,1))
        else:
            x = v / np.linalg.norm(v)
        
        # f_aff(x) = Bx + c
        x = B @ x + c

        return x

    def margin(self):
        return 0.02