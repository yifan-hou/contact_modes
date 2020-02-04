


class System(object):
    def __init__(self):
        self.collider = None
        self.bodies = []

    def set_collider(self, collider):
        self.collider = collider

    def add_body(self):
        pass

    def reindex_dof_masks(self):
        pass

    def num_dofs(self):
        pass

    def get_dofs(self):
        pass

    def set_dofs(self, q):
        pass

    def draw(self, shader):
        pass