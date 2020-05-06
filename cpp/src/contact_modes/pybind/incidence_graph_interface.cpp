#include <contact_modes/pybind/incidence_graph_interface.hpp>


// On creation, assumes ownership of Node*.
NodePython::NodePython(Node* node) {
    _node = node;
}

int NodePython::rank() {
    return _node->rank;
}

Eigen::VectorXd NodePython::interior_point() {
    return _node->interior_point;
}

Eigen::VectorXi NodePython::position() {
    return _node->position;
}

std::string     NodePython::sign_vector() {
    return _node->sign_vector;
}

std::vector<NodePythonPtr> NodePython::subfaces() {
    std::vector<NodePythonPtr> subs;
    for (Node* u : _node->subfaces) {
        subs.push_back(std::make_shared<NodePython>(u));
    }
    return subs;
}

std::vector<NodePythonPtr> NodePython::superfaces() {
    std::vector<NodePythonPtr> supers;
    for (Node* u : _node->superfaces) {
        supers.push_back(std::make_shared<NodePython>(u));
    }
    return supers;
}

IncidenceGraphPython::IncidenceGraphPython(IncidenceGraph* graph) {
    _graph = graph;
}

Eigen::MatrixXd IncidenceGraphPython::A() const {
    return _graph->A;
}

Eigen::VectorXd IncidenceGraphPython::b() const {
    return _graph->b;
}

int IncidenceGraphPython::dim() const {
    return _graph->dim();
}

int IncidenceGraphPython::num_k_faces(int k) const {
    // return _graph->num_k_faces(k);
    return -1;
}

int IncidenceGraphPython::num_incidences() const {
    return -1;
}

int IncidenceGraphPython::num_nodes() const {
    return _graph->_nodes.size();
}

NodePythonPtr IncidenceGraphPython::node(int i) {
    return std::make_shared<NodePython>(_graph->_nodes[i]);
}

std::vector<NodePythonPtr>   IncidenceGraphPython::rank(int k) {
    std::vector<NodePythonPtr> r;
    for (Node* u : _graph->rank(k)) {
        r.push_back(std::make_shared<NodePython>(u));
    }
    return r;
}

std::vector<Eigen::VectorXd> IncidenceGraphPython::interior_points() {
    std::vector<Eigen::VectorXd> int_pts;
    for (int i = 0; i <= _graph->dim(); i++) {
        for (Node* u : _graph->rank(i)) {
            int_pts.push_back(u->interior_point);
        }
    }
    return int_pts;
}

std::vector<Eigen::VectorXi> IncidenceGraphPython::positions() {
    std::vector<Eigen::VectorXi> positions;
    for (int i = 0; i <= _graph->dim(); i++) {
        for (Node* u : _graph->rank(i)) {
            positions.push_back(u->position);
        }
    }
    return positions;
}

void IncidenceGraphPython::update_sign_vectors(double eps) {
    _graph->update_sign_vectors(eps);
}

std::vector<std::string>     IncidenceGraphPython::sign_vectors() {
    std::vector<std::string> sign_vectors;
    for (int i = 0; i <= _graph->dim(); i++) {
        for (Node* u : _graph->rank(i)) {
            sign_vectors.push_back(u->sign_vector);
        }
    }
    return sign_vectors;
}