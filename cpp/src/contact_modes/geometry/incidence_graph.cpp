#include <contact_modes/geometry/incidence_graph.hpp>


Node::Node(int k) {
    this->rank = k;
    this->_color = 0;
    this->_black_bit = 0;
    this->_key.clear();
}

IncidenceGraph::IncidenceGraph(int d) 
    : _num_nodes_created(0)
{
    this->_lattice.resize(d + 3);
    for (int i = 0; i < d + 3; i++) {
        this->_lattice[i].clear();
    }
}

IncidenceGraph::~IncidenceGraph() {
}

int IncidenceGraph::dim() {
    return this->A.cols();
}

int IncidenceGraph::num_incidences() {
    return 0;
}

void IncidenceGraph::add_hyperplane(const Eigen::VectorXd& a, double d) {
    int n = this->A.rows();
    int m = this->A.cols();
    Eigen::MatrixXd A(n + 1, m);
    A << this->A, a.transpose();
    Eigen::VectorXd b(n + 1);
    b << this->b, d;
    this->A = A;
    this->b = b;
}

NodePtr IncidenceGraph::make_node(int k) {
    NodePtr node = std::make_shared<Node>(k);
    node->_id = this->_num_nodes_created++;
    node->_graph = shared_from_this();
    return node;
}

void IncidenceGraph::add_node(NodePtr node) {
    this->rank(node->rank).insert({node->_key, node->_id});
}

void IncidenceGraph::remove_node(NodePtr node) {
    // Remove arcs.
    for (int f_id : node->subfaces) {
        this->_nodes[f_id]->superfaces.erase(node->_id);
    }
    for (int g_id : node->superfaces) {
        this->_nodes[g_id]->subfaces.erase(node->_id);
        if (node->_color == COLOR_AH_GREY) {
            this->_nodes[g_id]->_grey_subfaces.erase(node->_id);
        }
        if (node->_color == COLOR_AH_BLACK) {
            this->_nodes[g_id]->_black_subfaces.erase(node->_id);
        }
    }
    // Remove node.
    this->rank(node->rank).erase(node->_key);
}

NodePtr IncidenceGraph::get_node(const std::string& key, int k) {
    auto iter = rank(k).find(key);
    if (iter == rank(k).end()) {
        return nullptr;
    } else {
        return this->_nodes[iter->second];
    }
}

Rank& IncidenceGraph::rank(int k) {
    return this->_lattice[k+1];
}