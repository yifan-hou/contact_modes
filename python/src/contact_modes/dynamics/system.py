import numpy as np
import quadprog

from contact_modes import (build_normal_velocity_constraints,
                           build_tangential_velocity_constraints)

DEBUG = False

class System(object):
    def __init__(self):
        self.collider = None
        self.bodies = []
        self.obstacles = []

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
        # 1. Run collision checker to get contact manifolds.
        manifolds = self.collider.collide()
        n_pts = len(manifolds)

        # 2. Create contact constraints.
        A, b = build_normal_velocity_constraints(manifolds)

        # 3. Create a cost function which minimizes distance with target
        #    velocity.
        n_dofs = A.shape[1]
        G = np.eye(n_dofs)
        a = qdot_star.T.copy()
        if DEBUG:
            print('qdot*')
            print(qdot_star.T)

        # 4. Convert normal velocity constraints to quadprog format, i.e. 
        #    Ax >= b.
        A *= -1
        b *= -1

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
        lamb1 = 1000
        G += lamb1 * C.T @ C
        a += lamb1 * b_C.T @ C

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
        C = S.T
        b = b_S.reshape((-1,))
        if list(b):
            sol = quadprog.solve_qp(G, a, C, b)
        else:
            sol = quadprog.solve_qp(G, a)
        x = sol[0].reshape((-1,1))

        if DEBUG:
            print(x)
            print(C.T @ x)
            print(C.T @ x - b.reshape((-1,1)))
            print(b.reshape((-1,1)))

        return x
