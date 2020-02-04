import numpy as np


class System(object):
    def __init__(self):
        self.collider = None
        self.bodies = []
        self.obstacles = []

    def set_collider(self, collider):
        self.collider = collider

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

    def draw(self, shader):
        for body in self.bodies:
            body.draw(shader)
        for obstacle in self.obstacles:
            obstacle.draw(shader)
