#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <contact_modes/geometry/incidence_graph.hpp>
#include <contact_modes/geometry/arrangements.hpp>
#include <contact_modes/collision/collide_2d.hpp>
#include <contact_modes/collision/manifold_2d.hpp>


namespace py = pybind11;

void exportGeometry(py::module& m) {
    py::class_<Node, NodePtr>(m, "Node")
        .def(py::init<int>())
        .def_readonly("rank", &Node::rank)
        .def_readwrite("interior_point", &Node::interior_point)
        .def_readonly("subfaces", &Node::subfaces)
        .def_readonly("superfaces", &Node::superfaces);

    py::class_<IncidenceGraph, IncidenceGraphPtr>(m, "IncidenceGraph")
        .def(py::init<int>())
        .def_readonly("A", &IncidenceGraph::A)
        .def_readonly("b", &IncidenceGraph::b)
        .def("dim", &IncidenceGraph::dim)
        .def("rank", &IncidenceGraph::rank)
        .def("get_node", &IncidenceGraph::get_node);

    m.def("initial_arrangement", &initial_arrangement);
}

void exportCollision(py::module& m) {

    py::class_<Manifold2D, Manifold2DPtr>(m, "Manifold2D")
        .def_readonly("pts_A", &Manifold2D::pts_A)
        .def_readonly("pts_B", &Manifold2D::pts_B)
        .def_readonly("normal", &Manifold2D::normal)
        .def_readonly("dists", &Manifold2D::dists);

    m.def("collide_2d", &collide_2d);
}