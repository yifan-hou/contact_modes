import numpy as np

from contact_modes import SE3, SO3

from .halfedgemesh import HalfedgeMesh


class Link(HalfedgeMesh):
    def __init__(self, r=0.5, l=1.5, outer_step=1.0):
        super(Link, self).__init__()

        # Create path + normals.
        n_ring = int(np.ceil(4*r*np.pi**2/outer_step))
        n_length = int(np.ceil(l/outer_step))
        path = np.zeros((3, n_ring * 2 + n_length * 2))
        normals = np.zeros((3, n_ring * 2 + n_length * 2))
        k = 0
        for i in range(n_length):
            path[0,k] = -l/2 + i/n_length*l
            path[1,k] = 2 * r
            normals[1,k] = 1.0
            k += 1
        for i in range(n_ring):
            theta = np.pi/2 - i/n_ring*np.pi
            path[0,k] = l/2 + np.cos(theta) * 2*r
            path[1,k] = np.sin(theta) * 2*r
            normals[0,k] = np.cos(theta)
            normals[1,k] = np.sin(theta)
            k += 1
        for i in range(n_length):
            path[0,k] = l/2 - i/n_length*l
            path[1,k] = -2 * r
            normals[1,k] = -1.0
            k += 1
        for i in range(n_ring):
            theta = -np.pi/2 - i/n_ring*np.pi
            path[0,k] = -l/2 + np.cos(theta) * 2*r
            path[1,k] = np.sin(theta) * 2*r
            normals[0,k] = np.cos(theta)
            normals[1,k] = np.sin(theta)
            k += 1
        
        # Create polygons and vertices.
        n_path = path.shape[1]
        n_loop = int(np.ceil(2*r*np.pi**2/outer_step))
        vertices = np.zeros((3, n_path, n_loop))
        indicies = np.array(range(n_path * n_loop)).reshape((n_path, n_loop))
        polygons = []
        for i in range(n_path + 1):
            if i < n_path:
                x = normals[:,i,None]
                for j in range(n_loop):
                    theta = 2*np.pi*j/n_loop
                    R = SO3.exp(theta*np.array([x[1,0], -x[0,0], 0.0]))
                    vertices[:,i,j,None] = (path[:,i,None] + 
                                            SO3.transform_point(R, r * x))
            if i == 0:
                continue
            idx_prev = indicies[i-1,:]
            idx_next = indicies[i%n_path,:]
            for j in range(1, n_loop + 1):
                poly = [idx_next[j-1], idx_prev[j-1], idx_prev[j%n_loop], idx_next[j%n_loop]]
                polygons.append(poly)
        vertices = vertices.reshape((3,n_path * n_loop))
        print(vertices.shape)
        vertices = [vertices[:,i,None] for i in range(vertices.shape[1])]
        print(len(vertices))

        # Build link.
        self.build(polygons, vertices)

    def init(self):
        self.reindex()
        self.set_tf_world(SE3.identity())

        self.init_geometry()
        self.init_opengl(True)