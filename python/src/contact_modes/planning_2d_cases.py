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

def box_to_shelf():
    box = Body2D('box')
    box.set_shape(Box2D())
    box.set_pose([0., -1., 0.])
    
    ground = Static2D('ground')
    shape = Box2D(6, 0.5)
    shape.draw_outline = False
    ground.set_shape(shape)
    ground.set_pose([0., -1.75, 0.])

    ceiling = Static2D('ceiling')
    shape = Box2D(6, 0.5)
    shape.draw_outline = False
    ceiling.set_shape(shape)
    ceiling.set_pose([0., 1.75, 0.])

    left_wall = Static2D('left-wall')
    shape = Box2D(0.5, 3)
    shape.draw_outline = False
    left_wall.set_shape(shape)
    left_wall.set_pose([-2.75,0.0,0])

    right_wall = Static2D('right-wall')
    shape = Box2D(0.5, 3)
    shape.draw_outline = False
    right_wall.set_shape(shape)
    right_wall.set_pose([2.75,0.0,0])

    shelf = Static2D('shelf')
    shape = Box2D(1.5, 0.25)
    shape.draw_outline = False
    shelf.set_shape(shape)
    shelf.set_pose([-1.75,0.125,0])

    collider = CollisionManager2D()
    collider.add_pair(box, ground)
    collider.add_pair(box, ceiling)
    collider.add_pair(box, left_wall)
    collider.add_pair(box, right_wall)
    collider.add_pair(box, shelf)
    collider.collide()

    system = System()
    system.add_body(box)
    system.add_obstacle(ground)
    system.add_obstacle(ceiling)
    system.add_obstacle(left_wall)
    system.add_obstacle(right_wall)
    system.add_obstacle(shelf)
    system.set_collider(collider)
    # system.reindex_dof_masks()

    return system