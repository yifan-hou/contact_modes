import numpy as np

from contact_modes.collision import CollisionManager2D, CollisionManifold2D
from contact_modes.dynamics import Body2D, Static2D, System
from contact_modes.shape import Polygon2D, Box2D


def box_ground():
    box = Body2D('box')
    box.set_shape(Box2D())
    box.set_collision_shape(Box2D())
    box.set_pose([0., 0., 0.])
    ground = Static2D('ground')
    ground.set_shape(Box2D(5))
    ground.set_collision_shape(Box2D(5.0, 1.0))
    ground.set_pose([0., -1, 0.])

    collider = CollisionManager2D()
    collider.add_pair(box, ground)
    collider.collide()

    system = System()
    system.add_body(box)
    system.add_obstacle(ground)
    system.set_collider(collider)
    # system.reindex_dof_masks()

    return system