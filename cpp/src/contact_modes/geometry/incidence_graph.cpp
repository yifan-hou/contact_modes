#include <contact_modes/geometry/incidence_graph.hpp>


Node::Node(int k) {
    this->_color = 0;
    this->rank = k;
    this->_black_bit = 0;
    this->_sv_key.clear();
}

IncidenceGraph::IncidenceGraph(int d) {
    this->_lattice.resize(d + 3);
    for (int i = 0; i < d + 3; i++) {
        this->_lattice[i].clear();
    }
}

IncidenceGraph::~IncidenceGraph() {
    for (int i = 0; i < this->_lattice.size(); i++) {
        auto iter = this->_lattice[i].begin();
        auto end = this->_lattice[i].end();
        while (iter != end) {
            NodePtr g = iter->second;
            g->subfaces.clear();
            g->superfaces.clear();
            g->_grey_subfaces.clear();
            g->_black_subfaces.clear();
            iter++;
        }
        this->_lattice[i].clear();
    }
}

int IncidenceGraph::dim() {
    return this->A.cols();
}

int IncidenceGraph::num_incidences() {
    return 0;
}

void IncidenceGraph::add_halfspace(const Eigen::VectorXd& a, double d) {
    int n = this->A.rows();
    int m = this->A.cols();
    Eigen::MatrixXd A(n + 1, m);
    A << this->A, a.transpose();
    Eigen::VectorXd b(n + 1);
    b << this->b, d;
    this->A = A;
    this->b = b;
}

void IncidenceGraph::add_node(const NodePtr& node) {
    this->rank(node->rank).insert({node->_sv_key, node});
}

void IncidenceGraph::remove_node(const NodePtr& node) {
    // Remove arcs.
    for (NodePtr f : node->subfaces) {
        f->superfaces.erase(node);
    }
    for (NodePtr g : node->superfaces) {
        g->subfaces.erase(node);
        if (node->_color == COLOR_AH_GREY) {
            g->_grey_subfaces.erase(node);
        }
        if (node->_color == COLOR_AH_BLACK) {
            g->_black_subfaces.erase(node);
        }
    }
    // Remove node.
    this->rank(node->rank).erase(node->_sv_key);
}

NodePtr IncidenceGraph::get_node(const std::string& sv_key, int k) {
    auto iter = rank(k).find(sv_key);
    if (iter == rank(k).end()) {
        return nullptr;
    } else {
        return iter->second;
    }
}

std::map<std::string, NodePtr>& IncidenceGraph::rank(int k) {
    return this->_lattice[k+1];
}