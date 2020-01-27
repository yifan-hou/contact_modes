from collada import *
from collada.polylist import Polylist
from collada.triangleset import TriangleSet

from contact_modes.shape import HalfedgeMesh


def import_mesh(dae_path, mesh):
    collada_mesh = Collada(dae_path)
    geom = collada_mesh.geometries[0]
    for primitive in geom.primitives:
        if isinstance(primitive, Polylist):
            polylist = primitive
            simplices = len(polylist)*[[]]
            for p, i in zip(polylist, range(len(polylist))):
                simplices[i] = p.indices.tolist()
            positions = polylist.vertex
            mesh.build(simplices, positions)
        elif isinstance(primitive, TriangleSet):
            triset = primitive
            simplices = len(triset)*[[]]
            for p, i in zip(triset, range(len(triset))):
                simplices[i] = p.indices.tolist()
            positions = triset.vertex
            mesh.build(simplices, positions)
        else:
            raise RuntimeError('Not implemented')
