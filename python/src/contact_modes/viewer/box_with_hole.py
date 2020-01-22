import matplotlib.pyplot as plt
import numpy as np

from .halfedgemesh import HalfedgeMesh

DEBUG = False

def smallfmod(x, y):
    while x > y:
        x -= y
    while x < 0:
        x += y
    return x

class BoxWithHole(HalfedgeMesh):
    def __init__(self, radius=0.5, side_length=1.0, height=1.0, n=30):
        super(BoxWithHole, self).__init__()
        # ----------------------------------------------------------------------
        # Create vertices
        # ----------------------------------------------------------------------
        theta = [i/n*2*np.pi for i in range(n)]
        theta.extend([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        theta = sorted(theta)
        n_t = len(theta)
        top_in_ids = range(0, n_t)
        top_out_ids = range(n_t, 2*n_t)
        bot_in_ids = range(2*n_t, 3*n_t)
        bot_out_ids = range(3*n_t, 4*n_t)
        top_side_ids = range(4*n_t, 4*n_t + 4)
        bot_side_ids = range(4*n_t + 4, 4*n_t + 8)

        top_in_verts = [None] * n_t
        top_out_verts = [None] * n_t
        bot_in_verts = [None] * n_t
        bot_out_verts = [None] * n_t
        top_side_verts = [None] * 4
        bot_side_verts = [None] * 4
        r = radius
        h = height
        l = side_length
        corners = [7*np.pi/4, np.pi/4, 3*np.pi/4, 5*np.pi/4]
        for i in range(n_t):
            t = theta[i]
            c = np.cos(t)
            s = np.sin(t)
            top_in_verts[i] = np.array([r*c, r*s,  h/2])
            bot_in_verts[i] = np.array([r*c, r*s, -h/2])

            for j in range(len(corners)):
                delta = smallfmod(t - corners[j], 2*np.pi)
                if delta <= np.pi/2:
                    # print(t, delta, j)
                    break
            if j == 0 or j == 2:
                c = l/2 if j == 0 else -l/2
                s = np.tan(t)*c
            if j == 1 or j == 3:
                s = l/2 if j == 1 else -l/2
                c = s/np.tan(t)
            top_out_verts[i] = np.array([c, s,  h/2])
            bot_out_verts[i] = np.array([c, s, -h/2])
        
        corners = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
        for i in range(4):
            t = corners[i]
            c = l/2 if np.cos(t) > 0 else -l/2
            s = l/2 if np.sin(t) > 0 else -l/2
            top_side_verts[i] = np.array([c, s, h/2])
            bot_side_verts[i] = np.array([c, s,-h/2])

        if DEBUG:        
            plt.plot(top_out_verts[0,:], top_out_verts[1,:], '-')
            plt.plot(top_in_verts[0,:], top_in_verts[1,:], '-')
            plt.axis('equal')
            plt.show()

        # ----------------------------------------------------------------------
        # Create polygons
        # ----------------------------------------------------------------------
        polygons = []
        
        for i in range(n_t):
            i0 = i
            i1 = (i+1) % n_t
            # top hole face
            p = [top_in_ids[i0], top_out_ids[i0], top_out_ids[i1], top_in_ids[i1]]
            polygons.append(p)
            # botton hole face
            p = [bot_in_ids[i0], bot_in_ids[i1], bot_out_ids[i1], bot_out_ids[i0]]
            polygons.append(p)
            # interior face
            p = [top_in_ids[i0], top_in_ids[i1], bot_in_ids[i1], bot_in_ids[i0]]
            polygons.append(p)

        # 4 side faces
        for i in range(4):
            i0 = i
            i1 = (i+1) % 4
            p = [top_side_ids[i0], bot_side_ids[i0], bot_side_ids[i1], top_side_ids[i1]]
            polygons.append(p)

        # Build mesh
        verts = []
        verts.extend(top_in_verts)
        verts.extend(top_out_verts)
        verts.extend(bot_in_verts)
        verts.extend(bot_out_verts)
        verts.extend(top_side_verts)
        verts.extend(bot_side_verts)

        self.build(polygons, verts)
