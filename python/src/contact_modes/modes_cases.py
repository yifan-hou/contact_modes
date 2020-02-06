import numpy as np
from numpy.linalg import norm

from contact_modes.collision import (CollisionManager, DynamicCollisionManager,
                                     TransformManager)
from contact_modes.dynamics import AnthroHand, Body, Proxy, Static, System
from contact_modes.shape import (Box, BoxWithHole, Cylinder, Ellipse,
                                 Icosphere, Torus)

from .modes_3d import contacts_to_half
from .lattice import FaceLattice
from .se3 import SE3
from .so3 import SO3
from .util import get_color


def box_case(walls=1):
    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    target = Box()
    target.get_tf_world().set_translation(np.array([0, 0, 0.0]))
    target_wireframe = Box(1.0, 1.0, 1.0)
    target_wireframe.set_color(get_color('black'))
    target.set_wireframe(target_wireframe)

    x = np.array([1., 0, 0])
    y = np.array([0, 1., 0])
    z = np.array([0, 0, 1.])
    N = [z, y, x, -x, -y, -z]
    T = [(x,y), (z,x), (y,z), (z,y), (x,z), (y,x)]
    obstacles = []
    for i in range(walls):
        n = N[i]
        box_dims = np.array([10., 10., 10.])
        box_dims[np.where(np.abs(n) > 0)] = 1.0
        box = Box(*box_dims)
        box.get_tf_world().set_translation(-n)
        obstacles.append(box)

    # --------------------------------------------------------------------------
    # Collision manager
    # --------------------------------------------------------------------------
    manager = CollisionManager()
    for i in range(walls):
        for j in range(4):
            manager.add_pair(None, obstacles[i])
    
    # --------------------------------------------------------------------------
    # Contact points, normals, and tangents
    # --------------------------------------------------------------------------
    points = np.zeros((3, 4 * walls))
    normals = np.zeros((3, 4 * walls))
    tangents = np.zeros((3, 4 * walls, 2))
    k = 0
    for i in range(walls):
        for t_x in [1, -1]:
            for t_y in [1, -1]:
                points[:,k] = -0.5 * (N[i] + t_x * T[i][0] + t_y * T[i][1])
                normals[:,k] = N[i]
                tangents[:,k,0] = T[i][0]
                tangents[:,k,1] = T[i][1]
                k += 1

    # print('A og')
    # print(contacts_to_half(points, normals)[0])

    # Try with new collision detection.
    target0 = Body('target')
    target0.set_shape(Box())
    N = [z, y, x, -x, -y, -z]
    obstacles0 = []
    for i in range(walls):
        n = N[i]
        wall = Static('wall%d' % i)
        box_dims = np.array([10., 10., 10.])
        box_dims[np.where(np.abs(n) > 0)] = 1.0
        wall.set_shape(Box(*box_dims))
        tf = wall.get_transform_world()
        tf.set_translation(-n)
        wall.set_transform_world(tf)
        obstacles0.append(wall)
    collider = DynamicCollisionManager()
    for i in range(walls):
        for t_x in [1, -1]:
            for t_y in [1, -1]:
                pt = -0.5 * (N[i] + t_x * T[i][0] + t_y * T[i][1])
                pt = pt.flatten()
                proxy = Proxy('proxy%+d%+d%+d' % (i, -t_x, -t_y))
                proxy.set_body(target0)
                proxy.set_shape(Icosphere(0.1, 0))
                proxy.set_transform_body(SE3.exp([pt[0], pt[1], pt[2], 0, 0, 0]))
                collider.add_pair(proxy, obstacles0[i])
    manifolds = collider.collide()
    # for m in manifolds:
    #     print(m)

    system = System()
    system.add_body(target0)
    for obs in obstacles0:
        system.add_obstacle(obs)
    system.set_collider(collider)
    system.reindex_dof_masks()

    return system

def peg_in_hole(n=8):
    # --------------------------------------------------------------------------
    # Contact points, normals, and tangents
    # --------------------------------------------------------------------------
    radius = 3.0
    height = 20.0
    side_length = 10.0
    cylinder_scale = 1/1.5

    top_points = np.zeros((3, n))
    top_normals = np.zeros((3, n))
    top_tangents = np.zeros((3, n, 2))
    bot_points = np.zeros((3, n))
    bot_normals = np.zeros((3, n))
    bot_tangents = np.zeros((3, n, 2))
    theta = np.array([i/n*2*np.pi for i in range(n)])
    r = radius
    h = height * cylinder_scale
    l = side_length
    for i in range(n):
        t = theta[i]
        c = np.cos(t)
        s = np.sin(t)
        
        top_points[:,i] = np.array([r*c, r*s, h/2.0])
        bot_points[:,i] = np.array([r*c, r*s,-h/2.0])

        top_normals[:,i] = np.array([-c, -s, 0])
        bot_normals[:,i] = np.array([-c, -s, 0])

        top_tangents[:,i,0] = np.array([0.0, 0.0, 1.0])
        bot_tangents[:,i,0] = np.array([0.0, 0.0, 1.0])

        top_tangents[:,i,1] = np.cross(top_normals[:,i], top_tangents[:,i,0])
        bot_tangents[:,i,1] = np.cross(bot_normals[:,i], bot_tangents[:,i,0])

    points = np.concatenate((top_points, bot_points), axis=1)
    normals = np.concatenate((top_normals, bot_normals), axis=1)
    tangents = np.concatenate((top_tangents, bot_tangents), axis=1)

    # --------------------------------------------------------------------------
    # Object and obstacle meshes
    # --------------------------------------------------------------------------
    # Try with new system.
    peg = Body('peg')
    peg.set_shape(Cylinder(radius, height * cylinder_scale))
    hole = Static('hole')
    hole.set_shape(BoxWithHole(radius, side_length, height))

    collider = DynamicCollisionManager()
    for i in range(points.shape[1]):
        p0 = Proxy('p%d+' % i)
        p1 = Proxy('p%d-' % i)

        p0.set_body(peg)
        p0.set_shape(Icosphere(0.1, 0))
        p0.set_margin(0.1)
        p0.set_num_dofs(6) # HACK
        p1.set_body(peg)
        p1.set_shape(Icosphere(0.1, 0))
        p1.set_margin(0.1)
        p0.set_num_dofs(0) # HACK

        xi0 = np.zeros((6,1))
        xi0[0:3,0] = points[:,i] + 0.1 * normals[:,i]
        p0.set_transform_body(SE3.exp(xi0))
        xi1 = np.zeros((6,1))
        xi1[0:3,0] = points[:,i] - 0.1 * normals[:,i]
        p1.set_transform_body(SE3.exp(xi1))

        collider.add_pair(p0, p1)
    manifolds = collider.collide()
    # for m in manifolds:
    #     print(m)

    system = System()
    system.add_body(peg)
    system.add_obstacle(hole)
    system.set_collider(collider)
    system.reindex_dof_masks()

    return system

def hand_football(fixed_ball=False):
    # Object and obstacles
    hand = AnthroHand('hand')
    # hand_dofs = [0.967, 0.772, 1.052, 0.882, 0.882, 0.882, 0.967, 0.772, 1.052, 2.041, -0.590]
    hand_dofs = [0.966, 0.771, 1.051, 0.881, 0.881, 0.881, 0.966, 0.771, 1.051, 2.042, -0.589]
    hand.set_state(hand_dofs)
    if fixed_ball:
        football = Static('football')
    else:
        football = Body('football')
    football.set_shape(Ellipse(10, 5, 5))
    football.set_transform_world(SE3.exp([0,2.5,5+0.6/2,0,0,0]))

    # Close hand around football.
    collider = DynamicCollisionManager()
    for link in hand.links:
        collider.add_pair(football, link)
    collider.collide()
    
    # Create system.
    system = System()
    system.add_body(hand)
    system.add_obstacle(football)
    system.set_collider(collider)
    system.reindex_dof_masks()

    return system

