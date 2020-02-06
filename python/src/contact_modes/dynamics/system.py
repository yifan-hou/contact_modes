import numpy as np
import quadprog

import imgui

from contact_modes import (build_normal_velocity_constraints,
                           build_tangential_velocity_constraints)

DEBUG = False

class System(object):
    def __init__(self):
        self.collider = None
        self.bodies = []
        self.obstacles = []

        self.init_tracking_gui()

    def set_collider(self, collider):
        self.collider = collider

    def get_collider(self):
        return self.collider

    def add_body(self, body):
        self.bodies.append(body)

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def num_dofs(self):
        num_dofs = 0
        for body in self.bodies:
            num_dofs += body.num_dofs()
        for obstacle in self.obstacles:
            num_dofs += obstacle.num_dofs()
        return num_dofs

    def reindex_dof_masks(self):
        total_dofs = self.num_dofs()
        k = 0
        for body in self.bodies:
            body_dofs = body.num_dofs()
            body.reindex_dof_mask(k, total_dofs)
            k += body_dofs
        for obstacle in self.obstacles:
            obstacle_dofs = obstacle.num_dofs()
            obstacle.reindex_dof_mask(k, total_dofs)
            k += obstacle_dofs

    def get_state(self):
        q = np.zeros((self.num_dofs(), 1))
        for body in self.bodies:
            q += body.get_state()
        for obstacle in self.obstacles:
            q += obstacle.get_state()
        return q

    def set_state(self, q):
        for body in self.bodies:
            body.set_state(q)
        for obstacle in self.obstacles:
            obstacle.set_state(q)

    def step(self, q_dot):
        for body in self.bodies:
            body.step(q_dot)
        for obstacle in self.obstacles:
            obstacle.step(q_dot)

    def draw(self, shader):
        for body in self.bodies:
            body.draw(shader)
        for obstacle in self.obstacles:
            obstacle.draw(shader)

    def init_tracking_gui(self):
        self.cs_modes = None
        self.qdot = None
        self.qdot_star = None
        self.lamb0 = 1.0
        self.lamb1 = 1000

    def draw_tracking_gui(self):
        manifolds = self.collider.get_manifolds()
        if self.cs_modes is None:
            self.cs_modes = ['?'] * len(manifolds)
        if self.qdot is None:
            self.qdot = np.zeros((len(manifolds),))
        if self.qdot_star is None:
            self.qdot_star = np.zeros((len(manifolds),))

        imgui.begin('tracking controls')

        imgui.columns(2, 'controls', border=True)
        for m, i in zip(manifolds, range(len(manifolds))):
            imgui.text(m.shape_A.name)
            imgui.same_line()
            imgui.text(m.shape_B.name)
            imgui.same_line()
            imgui.text('%+.6f' % m.dist)
            imgui.same_line()
            imgui.text(str(self.cs_modes[i]))
            imgui.same_line()
            imgui.text('%+.3f' % self.qdot[i])
            imgui.same_line()
            imgui.text('%+.3f' % self.qdot_star[i])
        imgui.next_column()
        changed, self.lamb0 = imgui.slider_float('lamb0', self.lamb0, 0.01, 1.0)
        changed, self.lamb1 = imgui.slider_float('lamb1', self.lamb1, 0.0, 1e6, power=2.0)
        imgui.end()

    def track_velocity(self, qdot_star, cs_modes, ss_modes=None):
        """
        Finds a generalized velocity q̇ as close as possible to q̇* that satifies
        the input contact constraints.

        Arguments:
            qdot_star {np.ndarray} -- nx1 target velocity
            cs_modes {[str]}       -- contacting/separating mode string ∈ {c,s}ⁿ
            ss_modes {[str]}       -- sliding/sticking mode string ∈ {-,0,+}²ⁿ

        Returns:
            np.ndarray -- tracking velocity
        """
        if DEBUG:
            print('tracking velocity')
        # 1. Run collision checker to get contact manifolds.
        manifolds = self.collider.collide()
        n_pts = len(manifolds)

        # 2. Create contact constraints.
        N, dists = build_normal_velocity_constraints(manifolds)
        assert(N.shape[0] == len(manifolds))

        if DEBUG:
            # sort by distance and print top-k constraints.
            idx = np.flip(np.argsort(dists.flatten())).flatten()
            print(idx)
            for i in range(3):
                print('dist, mode', cs_modes[idx[i]])
                print(dists[idx[i]])
                print('N')
                print(N[idx[i],:])

        # 3. Create a cost function which minimizes distance with target
        #    velocity.
        n_dofs = N.shape[1]
        G = self.lamb0 * np.eye(n_dofs)
        a = self.lamb0 * qdot_star.T.copy()
        if DEBUG:
            # print('qdot*')
            # print(qdot_star.T)
            print('cs modes', cs_modes)

        # 4. Convert normal velocity constraints to quadprog format, i.e. 
        #    Ax >= b.
        A = -N
        b = -dists

        # 5. Add soft equality constraints for contacting modes.
        c = np.where(cs_modes == 'c')[0]
        mask = np.zeros((n_pts,), dtype=bool)
        mask[c] = 1
        n_contacting = np.sum(mask)
        k = 0
        C = np.zeros((n_contacting, n_dofs))
        b_C = np.zeros((n_contacting,1))
        for i in range(n_pts):
            if mask[i]:
                C[k,:] = A[i,:]
                b_C[k,0] = b[i,0]
                k += 1
        G += self.lamb1 * C.T @ C
        a += self.lamb1 * b_C.T @ C

        # 6. Add inequality constraints for separating modes.
        n_separating = np.sum(~mask)
        S = np.zeros((n_separating, n_dofs))
        b_S = np.zeros((n_separating, 1))
        k = 0
        for i in range(n_pts):
            if ~mask[i]:
                S[k,:] = A[i,:]
                b_S[k,0] = b[i,0]
                k += 1

        # 7. Solve QP.
        a = a.reshape((-1,))
        C0 = S.T
        b0 = b_S.reshape((-1,))
        if list(b0):
            sol = quadprog.solve_qp(G, a, C0, b0)
        else:
            sol = quadprog.solve_qp(G, a)
        x = sol[0].reshape((-1,1))

        if DEBUG:
            # print(x)
            # print(C0.T @ x)
            # print(C0.T @ x - b0.reshape((-1,1)))
            print('dist')
            print(-b.reshape((-1,1)).T)

        # 8. Save info.
        self.cs_modes = cs_modes
        self.qdot = x.flatten()
        self.qdot_star = qdot_star.flatten()

        return x
