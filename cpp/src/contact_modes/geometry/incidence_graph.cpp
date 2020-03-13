#include <contact_modes/geometry/incidence_graph.hpp>
#include <iostream>


int get_sign(double v, double eps) {
    assert(eps > 0);
    if (v > eps) {
        return 1;
    } else if (v < -eps) {
        return -1;
    } else {
        return 0;
    }
}

void get_sign(const Eigen::VectorXd& v, Eigen::VectorXi& sv, double eps) {
    eps = abs(eps);
    sv.resize(v.size());
    for (int i = 0; i < v.size(); i++) {
        sv[i] = get_sign(v[i], eps);
    }
}

Node::Node(int k) {
    this->rank = k;
    this->_color = 0;
    this->_black_bit = 0;
    this->_key.clear();
}

void Node::update_position(double eps) {
    if (this->rank == -1 || this->rank == this->_graph->dim() + 1) {
        return;
    } else {
        Eigen::VectorXd res = _graph->A * this->interior_point - _graph->b;
        get_sign(res, this->position, eps);
    }
    // if (DEBUG) {
    //     std::cout << "rank: " << this->rank << std::endl;
    //     std::cout << _graph->A << std::endl;
    //     std::cout << _graph->b << std::endl;
    //     std::cout << "int pt\n" << this->interior_point << std::endl;
    // }
}

void Node::update_sign_vector(double eps) {
    update_position(eps);
    sign_vector.clear();
    for (int i = 0; i < this->position.size(); i++) {
        if (this->position[i] == 0) {
            sign_vector.push_back('0');
        } else if (this->position[i] == 1) {
            sign_vector.push_back('+');
        } else if (this->position[i] == -1) {
            sign_vector.push_back('-');
        } else {
            assert(false);
        }
    }
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

void IncidenceGraph::update_positions(double eps) {
    for (int i = 0; i < this->_nodes.size(); i++) {
        this->_nodes[i]->update_position(eps);
    }
}

void IncidenceGraph::update_sign_vectors(double eps) {
    for (int i = 0; i < this->_nodes.size(); i++) {
        this->_nodes[i]->update_sign_vector(eps);
    }
}

Positions IncidenceGraph::get_positions() {
    Positions P;
    for (int i = 0; i < this->_nodes.size(); i++) {
        if (this->_nodes[i]->position.size() != 0) {
            P.push_back(this->_nodes[i]->position);
        }
    }
    return P;
}

SignVectors IncidenceGraph::get_sign_vectors() {
    SignVectors S;
    for (int i = 0; i < this->_nodes.size(); i++) {
        if (!this->_nodes[i]->sign_vector.empty()) {
            S.push_back(this->_nodes[i]->sign_vector);
        }
    }
    return S;
}

NodePtr IncidenceGraph::make_node(int k) {
    NodePtr node = std::make_shared<Node>(k);
    node->_id = this->_nodes.size();
    node->_graph = shared_from_this();
    this->_nodes.push_back(node);
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